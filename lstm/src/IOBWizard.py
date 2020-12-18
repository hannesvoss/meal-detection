from datetime import datetime


class IOBWizard:
    """
    Diese Klasse dient als Berechnungsmodell für die aktuell wirkende Insulinmenge an einem gegebenen Timestamp.
    Aktuell wird zwischen den Insulintypen Humalog und Fiasp unterschieden. Weitere folgen.
    """
    insulin: str
    temp_basal: list
    bolus: list

    def __init__(self, temp_basal: list, bolus: list, insulin: str):
        self.temp_basal = temp_basal
        self.bolus = bolus
        self.insulin = insulin

    def get_iob(self, timestamp: datetime):
        print("IOB requested for: ", timestamp)
        # TODO calculate the iob and return it
        iob = 0.0
        if self.insulin == "Humalog":
            iob = 1.0
        elif self.insulin == "Fiasp":
            iob = 2.0
        return iob
