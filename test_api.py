import sys
import requests

img = sys.argv[1] if len(sys.argv) > 1 else input("Image path: ").strip()
url = "http://localhost:8000/predict"
with open(img, "rb") as f:
    r = requests.post(url, files={"file": (img, f, "image/jpeg")})
d = r.json()
print(f"\n  {d['class']}\033[0m  -  {d['confidence']*100:.1f}% risk level - {d['risk_level']}\n")