# OpenSteth 1.0 — Assembly Guide

> ⚠️ **This is an experimental research device. Not a medical device. Not validated for clinical use. Do not use on patients.**

Build a wireless digital stethoscope for **<$100 USD** in about 30 minutes.

---

## What You Need

| Item | Notes |
|------|-------|
| Used medical stethoscope | A used Littmann or similar. Check eBay/second-hand medical supply sites — ~$50 |
| DJI Mic Mini (transmitter + receiver) | ~$50. The transmitter is what you modify. |
| Wire cutters / side cutters | To cut the stethoscope tube |
| Small Phillips screwdriver | For the 3 DJI housing screws |
| Hairdryer or heat gun | To soften the DJI housing glue |
| Cutter / craft knife | To widen the housing opening |
| Small amount of glue | Superglue or hot glue |
| Cable ties (zip ties) | 2–3 small ones, to hold the housing closed during curing |

---

## Step 1 — Cut the Stethoscope Tube & Open the DJI Mic Mini

![Step 1](docs/assembly/1.png)

1. **Cut a short section of tubing** from the stethoscope — approximately 4–6 cm. Use wire cutters. You only need a small piece of the acoustic tubing from the stethoscope body.

2. **Open the DJI Mic Mini transmitter:**
   - Apply a hairdryer for **15–30 seconds** to the seam of the DJI transmitter housing. This softens the internal adhesive — do not overheat or you risk damaging the plastic or battery.
   - The housing can then be carefully pried open.
   - Inside you will find **3 small Phillips screws**. Remove and set them aside somewhere safe.
   - Carefully lift the PCB and battery out. Be gentle — the ribbon connectors are delicate.

---

## Step 2 — Fit the Microphone Into the Tube

![Step 2](docs/assembly/2.png)

1. **Test the fit:** The DJI Mic Mini microphone capsule (on the PCB) should slide snugly into the end of the stethoscope tube. The fit is naturally good — the tube diameter closely matches the capsule.

2. **Widen the housing opening:** Use a cutter or craft knife to carefully enlarge the small opening on the top face of the DJI housing — just enough for the stethoscope tube to pass through snugly. Work slowly; you want the tube to fit tightly, **not loosely**. The tube should seal the opening completely.

3. **Route the tube through the housing:** Thread the stethoscope tube through the opening so it sits over the microphone capsule inside.

4. **Reassemble:** Place the PCB and battery back into the housing. Replace the 3 screws. The tube should protrude from the top of the closed housing.

---

## Step 3 — Seal & Attach to Stethoscope

![Step 3](docs/assembly/3.png)

1. **Apply a small amount of glue** at the junction where the tube exits the housing to create an airtight seal. Superglue or hot glue both work.

2. **Use cable ties** to firmly clamp the housing closed while the glue cures. The tube is intentionally slightly wider than the housing opening — this creates the pressure needed to keep the seal airtight and prevents the tube from pulling out.

3. **Attach to the stethoscope chest piece:** Press the free end of the tube firmly onto the bell of the stethoscope chest piece. The tube acoustically couples the stethoscope's closed air column directly to the microphone capsule.

4. **Done.** The DJI transmitter clips onto clothing as usual. The stethoscope chest piece is pressed against the patient's chest. Lung sounds travel through the stethoscope's acoustic chamber, up the tube, and directly to the microphone capsule — then transmitted wirelessly via DJI to a connected phone.

---

## Total Cost

| Item | Approx. cost |
|------|-------------|
| Used medical stethoscope | ~$50 |
| DJI Mic Mini (transmitter + receiver) | ~$50 |
| **Total** | **<$100 USD** |

---

## Recording for OpenStethoscope

Once assembled:
1. Connect the DJI receiver to your phone (USB-C or Lightning adapter)
2. To simply record lung sounds, open any audio recording app
   — for live monitoring or AI-assisted classification, start a local instance of OpenStethoscope or open **[opensteth.agentic-medicine.com](https://opensteth.agentic-medicine.com)**
3. Record lung sounds
4. If desired: upload the recording, embed and analyse it, and receive results

---

> **Disclaimer:** OpenSteth 1.0 is an experimental research tool built for the Google MedGemma Impact Challenge (Feb 2026). It is not a certified medical device, has not been approved by any regulatory authority, and must not be used for clinical decision-making. Always consult a qualified healthcare professional.
