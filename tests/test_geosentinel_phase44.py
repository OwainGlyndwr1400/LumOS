"""GeoSentinel Phase 44 — pure-transform tests for the new world-sensing feeds.

All offline: each new feed splits its network fetch from a PURE parse/logic
function (parse_gdacs_xml, parse_firms_csv, fresh_disaster_trips, map_zones,
parse_tle_text, parse_satcat_csv, parse_ioda_events, fresh_outage_trips,
describe_satellite). We exercise those directly with canned data — no feed hits,
matching the repo's test discipline (test_rail_alert_filter etc.).
"""

from lumos_node.telemetry import fires, gdacs, outages, satellites
from lumos_node.telemetry.conflict import map_zones

# ── GDACS ────────────────────────────────────────────────────────────────────

_GDACS_XML = """<?xml version="1.0"?>
<rss xmlns:gdacs="http://www.gdacs.org" xmlns:georss="http://www.georss.org/georss">
<channel>
  <item>
    <title>Red alert Tropical Cyclone for FIJI</title>
    <link>https://www.gdacs.org/report.aspx?eventid=1000</link>
    <gdacs:eventid>1000</gdacs:eventid>
    <gdacs:eventtype>TC</gdacs:eventtype>
    <gdacs:alertlevel>Red</gdacs:alertlevel>
    <gdacs:country>Fiji</gdacs:country>
    <georss:point>-17.7 178.0</georss:point>
  </item>
  <item>
    <title>Green earthquake M5.1</title>
    <gdacs:eventid>1001</gdacs:eventid>
    <gdacs:eventtype>EQ</gdacs:eventtype>
    <gdacs:alertlevel>Green</gdacs:alertlevel>
    <gdacs:country>Chile</gdacs:country>
    <georss:point>-30.0 -71.0</georss:point>
  </item>
</channel>
</rss>"""


def test_gdacs_parse_extracts_events_and_levels():
    events = gdacs.parse_gdacs_xml(_GDACS_XML)
    assert len(events) == 2
    red = next(e for e in events if e["event_id"] == "1000")
    assert red["level"] == "red" and red["level_rank"] == 3
    assert red["type_label"] == "Tropical cyclone"
    assert red["lat"] == -17.7 and red["lon"] == 178.0


def test_gdacs_rejects_dtd_xxe():
    malicious = '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY x "y">]><rss></rss>'
    assert gdacs.parse_gdacs_xml(malicious) == []


def test_gdacs_fresh_trips_new_then_silent_then_escalation():
    events = gdacs.parse_gdacs_xml(_GDACS_XML)
    # min_rank=2 (orange+): only the red cyclone is at/above; green EQ is recorded
    # but not tripped.
    trips, seen = gdacs.fresh_disaster_trips(events, {}, min_rank=2, operator_lat=51.6, operator_lon=-4.0)
    assert [t["data"]["event_id"] for t in trips] == ["1000"]
    assert seen["1001"] == 1  # recorded for future escalation detection
    # Same events again → nothing new.
    trips2, seen2 = gdacs.fresh_disaster_trips(events, seen, min_rank=2, operator_lat=51.6, operator_lon=-4.0)
    assert trips2 == []
    # Green EQ escalates to orange → one fresh trip.
    events[1]["level"], events[1]["level_rank"] = "orange", 2
    trips3, _ = gdacs.fresh_disaster_trips(events, seen2, min_rank=2, operator_lat=51.6, operator_lon=-4.0)
    assert [t["data"]["event_id"] for t in trips3] == ["1001"]
    assert "escalated" in trips3[0]["description"]


# ── FIRMS fires ───────────────────────────────────────────────────────────────

_FIRMS_MODIS = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_t31,frp,daynight\n"
    "51.70,-4.05,320.1,1.0,1.0,2026-07-15,1312,Terra,MODIS,85,6.1,290.0,45.2,D\n"
    "51.72,-4.02,300.0,1.0,1.0,2026-07-15,1312,Terra,MODIS,12,6.1,285.0,5.0,D\n"
    "0.0,0.0,300.0,1.0,1.0,2026-07-15,1312,Terra,MODIS,90,6.1,285.0,5.0,D\n"
)


def test_firms_parse_normalizes_confidence_and_drops_null_island():
    rows = fires.parse_firms_csv(_FIRMS_MODIS)
    assert len(rows) == 2  # (0,0) row dropped
    assert rows[0]["confidence"] == "high"      # 85 → high
    assert rows[1]["confidence"] == "low"       # 12 → low
    assert rows[0]["frp_mw"] == 45.2
    assert rows[0]["acq_time_utc"] == "13:12"


def test_firms_viirs_letter_confidence():
    viirs = (
        "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        "instrument,confidence,version,bright_ti5,frp,daynight\n"
        "51.70,-4.05,330.0,0.4,0.4,2026-07-15,0201,N,VIIRS,n,2.0,295.0,12.0,N\n"
    )
    rows = fires.parse_firms_csv(viirs)
    assert rows[0]["confidence"] == "nominal"


def test_firms_cluster_cell_dedups_nearby():
    assert fires.cluster_cell(51.701, -4.052) == fires.cluster_cell(51.704, -4.049)


# ── Conflict zones ────────────────────────────────────────────────────────────

def test_conflict_map_zones_geocodes_and_counts():
    scored = [
        {"title": "Heavy shelling near Kharkiv as Ukraine war grinds on", "severity": 6, "source": "BBC"},
        {"title": "Kyiv struck by missile barrage overnight", "severity": 5, "source": "AJ"},
        {"title": "Gaza airstrike kills dozens", "severity": 7, "source": "AJ"},
        {"title": "Local council debates parking", "severity": 0, "source": "Local"},
    ]
    zones = map_zones(scored)
    ids = {z["id"] for z in zones}
    assert "ukraine" in ids and "gaza" in ids
    uk = next(z for z in zones if z["id"] == "ukraine")
    assert uk["event_count"] == 2 and uk["sum_severity"] == 11
    assert uk["lat"] == 48.5


# ── Satellites: TLE + SATCAT ──────────────────────────────────────────────────

_TLE_TEXT = """ISS (ZARYA)
1 25544U 98067A   26196.50000000  .00016717  00000-0  10270-3 0  9008
2 25544  51.6400 208.0000 0006703 130.0000 325.0000 15.72125391000000
STARLINK-1007
1 44713U 19074A   26196.50000000  .00001234  00000-0  10270-4 0  9999
2 44713  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000000000
"""


def test_parse_tle_text_reads_three_line_sets():
    tles = satellites.parse_tle_text(_TLE_TEXT)
    assert len(tles) == 2
    assert tles[0]["name"] == "ISS (ZARYA)"
    assert tles[0]["line1"].startswith("1 25544U")
    assert tles[1]["name"] == "STARLINK-1007"


def test_parse_tle_text_resyncs_past_stray_lines():
    junk = "GARBAGE HEADER LINE\n" + _TLE_TEXT
    assert len(satellites.parse_tle_text(junk)) == 2


def test_parse_satcat_maps_country_and_type():
    csv = (
        "OBJECT_NAME,OBJECT_ID,NORAD_CAT_ID,OBJECT_TYPE,OPS_STATUS_CODE,OWNER,LAUNCH_DATE\n"
        "ISS (ZARYA),1998-067A,25544,PAY,+,ISS,1998-11-20\n"
        "STARLINK-1007,2019-074A,44713,PAY,+,US,2019-11-11\n"
    )
    cat = satellites.parse_satcat_csv(csv)
    assert cat["44713"]["country"] == "USA"
    assert cat["44713"]["object_type"] == "payload"
    assert cat["44713"]["launch_date"] == "2019-11-11"
    assert cat["44713"]["intl_designator"] == "2019-074A"


def test_describe_satellite_builds_googleable_line():
    st = {
        "name": "STARLINK-1007", "norad": "44713", "country": "USA",
        "mission": "comms_constellation", "object_type": "payload", "launch_date": "2019-11-11",
    }
    desc = satellites.describe_satellite(st)
    assert "STARLINK-1007" in desc and "USA" in desc
    assert "launched 2019" in desc and "NORAD 44713" in desc


# ── IODA outages ──────────────────────────────────────────────────────────────

def test_parse_ioda_events_extracts_country_code():
    data = {"data": [
        {"location": "country/SD", "score": 120, "severity": "critical", "start": 1000, "duration": 3600, "datasource": "bgp_ping"},
        {"location": "country/IR", "score": 40, "severity": "minor", "start": 1100, "duration": 600},
    ]}
    evs = outages.parse_ioda_events(data)
    assert evs[0]["country_code"] == "SD"  # highest score first
    assert evs[0]["datasource"] == "bgp ping"


def test_outage_fresh_trips_respects_watchlist_and_score():
    evs = outages.parse_ioda_events({"data": [
        {"location": "country/SD", "score": 120, "severity": "critical", "start": 1000},
        {"location": "country/FR", "score": 30, "severity": "minor", "start": 1100},
    ]})
    # Watch GB only → nothing.
    assert outages.fresh_outage_trips(evs, set(), {"GB"}, min_score=50) == []
    # Any country, score >= 50 → only SD.
    trips = outages.fresh_outage_trips(evs, set(), None, min_score=50)
    assert [t["data"]["country_code"] for t in trips] == ["SD"]
    # Already seen → silent.
    assert outages.fresh_outage_trips(evs, {"SD-1000"}, None, min_score=50) == []
