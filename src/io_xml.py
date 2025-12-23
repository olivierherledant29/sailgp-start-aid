import xml.etree.ElementTree as ET
from pathlib import Path


#DEFAULT_XML_NAME = "25113006_10-40-01.xml"
DEFAULT_XML_NAME = "sample.xml"



def load_xml_bytes(uploaded, use_default: bool):
    if uploaded is not None:
        return uploaded.read()

    if use_default:
        p = Path(DEFAULT_XML_NAME)
        if p.exists():
            return p.read_bytes()

    return None


def parse_race_xml(xml_bytes: bytes):
    root = ET.fromstring(xml_bytes)

    def get_mark_latlon(name: str):
        for cm in root.iter("CompoundMark"):
            if cm.attrib.get("Name") == name:
                mk = cm.find("Mark")
                if mk is not None and "TargetLat" in mk.attrib and "TargetLng" in mk.attrib:
                    return (float(mk.attrib["TargetLat"]), float(mk.attrib["TargetLng"]))

        for mk in root.iter("Mark"):
            if mk.attrib.get("Name") == name and "TargetLat" in mk.attrib and "TargetLng" in mk.attrib:
                return (float(mk.attrib["TargetLat"]), float(mk.attrib["TargetLng"]))

        raise ValueError(f"Mark '{name}' not found in XML.")

    marks = {"SL1": get_mark_latlon("SL1"), "SL2": get_mark_latlon("SL2"), "M1": get_mark_latlon("M1")}

    course_limit = root.find("CourseLimit")
    if course_limit is None:
        raise ValueError("CourseLimit not found in XML.")

    limits = []
    for lim in course_limit.findall("Limit"):
        limits.append((float(lim.attrib["Lat"]), float(lim.attrib["Lon"])))

    if len(limits) < 3:
        raise ValueError("Boundary polygon (CourseLimit/Limit) has < 3 points.")

    return marks, limits
