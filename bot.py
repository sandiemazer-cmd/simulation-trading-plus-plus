"""
Robot de trading FICTIF : 1000 € virtuels, bougies 15 min, 24h/24.
Aucun ordre réel n'est passé : il lit seulement les prix publics de Binance.
Stratégie : EMA 50 (tendance) + croisement EMA 9/21 + MACD + RSI 50-70
+ volume, stop-loss et objectif calculés avec l'ATR. Levier x2 seulement
sur les signaux les plus forts.
"""
import csv
import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# ---------- Réglages ----------
URL = "https://data-api.binance.vision/api/v3/klines"  # données publiques Binance
PAIRES = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT"]
INTERVALLE = "15m"
CAPITAL_DEPART = 1000.0
FRAIS = 0.001          # 0,1 % par ordre (achat et vente)
SL_ATR = 1.0           # stop-loss = 1 x ATR sous le prix d'entrée
TP_ATR = 2.0           # objectif = 2 x ATR au-dessus
MAX_BOUGIES = 12       # sortie forcée après 3 h si ni stop ni objectif
HEURE_RECAP = 21       # récap quotidien Telegram vers 21 h (heure de Paris)
PARIS = ZoneInfo("Europe/Paris")
FICHIER_ETAT = "state.json"
FICHIER_TRADES = "trades.csv"
TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")


def telegram(msg):
    print(msg)
    if not TOKEN or not CHAT:
        print("⚠️ Telegram non configuré : secret TELEGRAM_TOKEN ou TELEGRAM_CHAT_ID_TRADING manquant")
        return
    try:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                          data={"chat_id": CHAT, "text": msg}, timeout=10)
        if not r.ok:
            print("⚠️ Telegram a refusé le message :", r.status_code, r.text)
    except Exception as e:
        print("⚠️ Erreur Telegram :", e)


def bougies(paire):
    r = requests.get(URL, params={"symbol": paire, "interval": INTERVALLE, "limit": 200}, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json(), columns=["t", "o", "h", "l", "c", "v", "ct", "q", "n", "tb", "tq", "i"])
    for k in ["o", "h", "l", "c", "v"]:
        df[k] = df[k].astype(float)
    maintenant = int(time.time() * 1000)
    return df[df["ct"] < maintenant].reset_index(drop=True)  # bougies terminées uniquement


def indicateurs(df):
    c = df["c"]
    df["ema9"] = c.ewm(span=9, adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    df["macdh"] = macd - macd.ewm(span=9, adjust=False).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    perte = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / perte)
    tr = pd.concat([df["h"] - df["l"], (df["h"] - c.shift()).abs(), (df["l"] - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    df["volmoy"] = df["v"].rolling(20).mean()
    return df


def signal(df):
    d, prec = df.iloc[-1], df.iloc[-2]
    e9, e21 = df["ema9"], df["ema21"]
    # croisement EMA 9 au-dessus de l'EMA 21 sur l'une des 3 dernières bougies
    croise = [e9.iloc[-i - 1] <= e21.iloc[-i - 1] and e9.iloc[-i] > e21.iloc[-i] for i in (1, 2, 3)]
    ok = (d.c > d.ema50 and d.ema9 > d.ema21 and any(croise)
          and d.macdh > 0 and 50 < d.rsi < 70 and d.v > d.volmoy)
    if not ok:
        return None
    fort = croise[0] and d.v > 1.5 * d.volmoy and d.macdh > prec.macdh
    return {"score": d.v / d.volmoy, "levier": 2 if fort else 1}


def charger():
    if os.path.exists(FICHIER_ETAT):
        with open(FICHIER_ETAT) as f:
            return json.load(f)
    return {"capital": CAPITAL_DEPART, "position": None, "jour": None, "trades_jour": 0,
            "gagnants_jour": 0, "recap_envoye": None, "signaux_vus": {},
            "debut": dt.datetime.now(PARIS).isoformat(timespec="minutes")}


def enregistrer_trade(ligne):
    nouveau = not os.path.exists(FICHIER_TRADES)
    with open(FICHIER_TRADES, "a", newline="") as f:
        w = csv.writer(f)
        if nouveau:
            w.writerow(["entree_le", "sortie_le", "paire", "levier", "prix_entree", "prix_sortie",
                        "raison", "resultat_eur", "capital"])
        w.writerow(ligne)


def fermer(s, prix, raison, heure):
    p = s["position"]
    expo = p["mise"] * p["levier"]
    brut = expo * (prix / p["entree"] - 1)
    frais = expo * FRAIS + expo * (prix / p["entree"]) * FRAIS
    res = brut - frais
    s["capital"] = round(s["capital"] + res, 2)
    s["trades_jour"] += 1
    if res > 0:
        s["gagnants_jour"] += 1
    enregistrer_trade([p["heure"], heure, p["paire"], p["levier"], p["entree"], round(prix, 6),
                       raison, round(res, 2), s["capital"]])
    emoji = "✅" if res > 0 else "❌"
    telegram(f"{emoji} [SIMU] Vente {p['paire']} ({raison})\n"
             f"Résultat : {res:+.2f} €\nCapital : {s['capital']:.2f} €")
    s["position"] = None


def gerer_position(s):
    p = s["position"]
    df = bougies(p["paire"])
    for _, b in df[df["t"] > p["derniere_bougie"]].iterrows():
        heure = dt.datetime.fromtimestamp(b.ct / 1000, PARIS).isoformat(timespec="minutes")
        p["bougies"] += 1
        p["derniere_bougie"] = int(b.t)
        if b.l <= p["sl"]:                      # prudence : stop vérifié en premier
            return fermer(s, p["sl"], "stop-loss", heure)
        if b.h >= p["tp"]:
            return fermer(s, p["tp"], "objectif atteint", heure)
        if p["bougies"] >= MAX_BOUGIES:
            return fermer(s, b.c, "temps écoulé", heure)


def chercher_entree(s):
    candidats = []
    for paire in PAIRES:
        try:
            df = indicateurs(bougies(paire))
        except Exception as e:
            print(paire, "erreur :", e)
            continue
        sig = signal(df)
        if sig and int(df["t"].iloc[-1]) > s["signaux_vus"].get(paire, 0):
            candidats.append((sig, paire, df.iloc[-1]))
    if not candidats:
        return
    sig, paire, d = max(candidats, key=lambda x: x[0]["score"])
    s["signaux_vus"][paire] = int(d.t)
    s["position"] = {
        "paire": paire, "entree": float(d.c), "levier": sig["levier"], "mise": s["capital"],
        "sl": float(d.c - SL_ATR * d.atr), "tp": float(d.c + TP_ATR * d.atr),
        "derniere_bougie": int(d.t), "bougies": 0,
        "heure": dt.datetime.fromtimestamp(d.ct / 1000, PARIS).isoformat(timespec="minutes"),
    }
    p = s["position"]
    telegram(f"🟢 [SIMU] Achat {paire} à {p['entree']:.6g} $ (levier x{p['levier']})\n"
             f"Stop : {p['sl']:.6g} | Objectif : {p['tp']:.6g}\nMise : {p['mise']:.2f} €")


def recap(s):
    maintenant = dt.datetime.now(PARIS)
    if maintenant.hour >= HEURE_RECAP and s["recap_envoye"] != s["jour"]:
        perf = (s["capital"] / CAPITAL_DEPART - 1) * 100
        telegram(f"📊 [SIMU] Récap du {s['jour']}\n"
                 f"Trades : {s['trades_jour']} (gagnants : {s['gagnants_jour']})\n"
                 f"Capital : {s['capital']:.2f} € ({perf:+.1f} % depuis le début)")
        s["recap_envoye"] = s["jour"]


def main():
    s = charger()
    if os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":   # lancement manuel = test
        etat = f"en position sur {s['position']['paire']}" if s["position"] else "sans position"
        telegram(f"🔧 [SIMU] Bot en ligne, {etat}\nCapital : {s['capital']:.2f} €")
    aujourd_hui = dt.datetime.now(PARIS).date().isoformat()
    if s["jour"] != aujourd_hui:
        s["jour"], s["trades_jour"], s["gagnants_jour"] = aujourd_hui, 0, 0
    if s["position"]:
        try:
            gerer_position(s)
        except Exception as e:
            print("Erreur suivi position :", e)
    if not s["position"]:
        chercher_entree(s)
    recap(s)
    with open(FICHIER_ETAT, "w") as f:
        json.dump(s, f, indent=2)
    if not os.path.exists(FICHIER_TRADES):   # pour que GitHub ait toujours le fichier
        with open(FICHIER_TRADES, "w", newline="") as f:
            csv.writer(f).writerow(["entree_le", "sortie_le", "paire", "levier", "prix_entree",
                                    "prix_sortie", "raison", "resultat_eur", "capital"])


if __name__ == "__main__":
    main()
