"""Central settings. Edit LOCATIONS to add your own river points."""
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW, PROC, MODELS, REPORTS = (ROOT / "data" / "raw", ROOT / "data" / "processed",
                              ROOT / "models", ROOT / "reports")

START_DATE = "1990-01-01"                                   # GloFAS reanalysis starts 1984
END_DATE = (date.today() - timedelta(days=7)).isoformat()   # archive lags a few days
TRAIN_END, VAL_END = "2015-12-31", "2018-12-31"             # train <=2015 | val 2016-18 | test >=2019
TZ = "Asia/Kolkata"

# slug: (name, latitude, longitude) - points on/near large rivers (GloFAS snaps to the
# largest river within ~5 km). Coordinates are approximate; fetch_data.py skips points
# with no river coverage. Mix of flood-prone and dry places on purpose.
LOCATIONS = {
    "chennai": ("Chennai (Adyar)", 13.010, 80.270),
    "cuddalore": ("Cuddalore", 11.748, 79.771),
    "trichy": ("Tiruchirappalli (Cauvery)", 10.830, 78.690),
    "thanjavur": ("Thanjavur", 10.787, 79.138),
    "madurai": ("Madurai (Vaigai)", 9.925, 78.120),
    "vellore": ("Vellore (Palar)", 12.917, 79.133),
    "aluva": ("Aluva (Periyar)", 10.100, 76.357),
    "thrissur": ("Thrissur", 10.528, 76.214),
    "hyderabad": ("Hyderabad (Musi)", 17.385, 78.487),
    "vijayawada": ("Vijayawada (Krishna)", 16.506, 80.648),
    "rajahmundry": ("Rajahmundry (Godavari)", 17.000, 81.804),
    "cuttack": ("Cuttack (Mahanadi)", 20.463, 85.883),
    "kolkata": ("Kolkata (Hooghly)", 22.573, 88.364),
    "patna": ("Patna (Ganga)", 25.594, 85.138),
    "varanasi": ("Varanasi (Ganga)", 25.318, 82.974),
    "lucknow": ("Lucknow (Gomti)", 26.847, 80.946),
    "kanpur": ("Kanpur (Ganga)", 26.450, 80.332),
    "delhi": ("Delhi (Yamuna)", 28.660, 77.250),
    "guwahati": ("Guwahati (Brahmaputra)", 26.145, 91.736),
    "dibrugarh": ("Dibrugarh (Brahmaputra)", 27.473, 94.912),
    "srinagar": ("Srinagar (Jhelum)", 34.084, 74.797),
    "surat": ("Surat (Tapi)", 21.170, 72.831),
    "vadodara": ("Vadodara", 22.307, 73.181),
    "pune": ("Pune (Mula-Mutha)", 18.520, 73.857),
    "kolhapur": ("Kolhapur (Panchganga)", 16.705, 74.243),
    "jaipur": ("Jaipur (dry region)", 26.912, 75.787),
}
