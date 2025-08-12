import os
import xml.etree.ElementTree as ET


annotations_folder = r"D:\projects\vehicle_detection\data\annotations"


for xml_file in os.listdir(annotations_folder):
    if not xml_file.endswith(".xml"):
        continue  

    xml_path = os.path.join(annotations_folder, xml_file)

   
    if os.path.getsize(xml_path) == 0:
        print(f" Empty file removed: {xml_file}")
        os.remove(xml_path)
        continue

 
    try:
        ET.parse(xml_path)
    except ET.ParseError as e:
        print(f" Corrupted XML removed: {xml_file} | Error: {e}")
        os.remove(xml_path)

print(" XML cleanup completed successfully!")
