# LSTM Versuch

## Datenquelle & Beschreibung der vorliegenden Daten
Es gibt drei relevante csv-Dateien die das Preprocessing durchlaufen müssen.
1. CGM Datei
2. Treatments Datei
3. Profile Datei

Aus dem CGM File sind der Timestamp in Kombination mit dem SGV (Glukosemesswert) relevant.
Aus der Treatments Datei sind die Insulinabgaben relevant. Diese sind in der einfachen Form als Event-Type 'Meal Bolus' 
gekennzeichnet und in der schwierigeren Form als 'Temp Basal'. 

Die Meal Boli haben das Feld 'insulin' in welchem die abgegebene Insulinmenge als Float Value enthalten sind.

Die Temp Basal Einträge haben das Feld 

## Bereinigung & Vorbereitung der Daten
Die Daten müssen von möglichen Fehlern (z.B. Ausreißer) und Perioden in denen Messwerte fehlen bereinigt werden. Erst danach 
kann mit den Daten weiter gearbeitet werden.