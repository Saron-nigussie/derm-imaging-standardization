# Smartphone Dermatological Imaging Standardization System

**Institution:** Florida International University  
**Department:** Biomedical Engineering  
**Project type:** Independent undergraduate research  
**Started:** May 2025  
**Status:** Physics foundation study + computational pipeline operational — pre-prototype phase


## The Problem

Smartphone dermatological images are inconsistent due to lighting 
variability, distance changes, camera angle differences, and ambient 
light interference. This inconsistency creates measurement noise larger 
than the biological signal of interest, making reproducible erythema 
and melanin quantification impossible with uncontrolled smartphone capture.

State-of-the-art dermatology AI models achieving AUC of 0.88–0.94 on 
controlled test sets drop to 0.56–0.65 on diverse real-world images 
(Daneshjou et al. 2022). This performance collapse happens at the 
acquisition layer, before any AI model processes the image.

## The Solution

A 3D-printed smartphone imaging attachment that controls the four dominant 
sources of acquisition variability:

- **Illumination** — 16-LED NeoPixel ring with opal acrylic diffuser
- **Working distance** — fixed 20mm contact ring
- **Camera angle** — enclosed barrel enforces perpendicular alignment
- **Ambient light** — fully enclosed matte black barrel

Primary biological targets: erythema (CIELAB a* channel) and melanin 
(CIELAB L* channel) quantification across Fitzpatrick Scale Types I–VI.

## Comparison to Prior Work

Closest published system: Uthoff et al. 2020 (J. Biomed. Opt. 25(6)).  
Key improvements in this system:

- Diffuse opal acrylic illumination (Uthoff had no diffuser)
- Multi-phone adjustable clip (Uthoff: LG G5 only, discontinued)
- Full ColorChecker color calibration (Uthoff: 18% gray card only)
- Explicit FST-stratified validation (Uthoff: not addressed)
- Open-source design release (Uthoff: not released)

## Repository Structure

| Folder | Contents |
|--------|----------|
| `/cad` | Fusion 360 design files (.f3d, .stl exports) |
| `/hardware` | Bill of materials, wiring diagrams, assembly guide |
| `/software/calibration` | Flatfield correction, color calibration scripts |
| `/software/metrics` | SSIM, blur, CoV, deltaE computation |
| `/data/phantom` | Phantom study captures and results |
| `/data/calibration` | White reference and ColorChecker captures |
| `/docs/literature_notes` | Paper summaries and annotations |
| `/docs/physics_notes` | Skin optics derivations and calculations |
| `/docs/lab_notebook` | Dated experimental records |
| `/docs/sketches` | Concept sketches and design iterations |

## Target Performance Metrics

| Metric | Target | Uthoff Baseline |
|--------|--------|-----------------|
| Illumination uniformity | > 0.960 | 0.954 |
| Color accuracy ΔE | < 3.0 | Not measured |
| Inter-session SSIM | Significant improvement vs bare phone | Not measured |
| FST consistency | Uniform across FST I–VI | Not tested |

## Bill of Materials — V1 Estimate

| Component | Specification | Est. Cost |
|-----------|--------------|-----------|
| PLA filament (matte black) | eSUN Matte PLA+ 1.75mm | ~$22 |
| TPU filament (black) | PolyFlex TPU95 1.75mm | ~$20 |
| LED ring | NeoPixel 16-RGBW 44mm 5V | ~$10 |
| Arduino Nano | ATmega328, 5V | ~$5 |
| Opal acrylic diffuser | White cast acrylic 2mm | ~$14 |
| Linear polarizer film | 40–45mm linear, two sheets | ~$14 |
| M2 heat-set inserts | Brass M2×4mm, qty 20 | ~$7 |
| M2 screws | Socket head cap, qty 20 | ~$8 |
| USAF resolution chart | 1951 standard target | ~$15 |
| 18% gray card | Matte photography reference | ~$8 |
| **Total V1 estimate** | | **~$123** |

## Key References

1. Uthoff et al. (2020). Point-of-care, multispectral, smartphone-based 
   dermascopes. *J. Biomed. Opt.* 25(6).
   DOI: 10.1117/1.JBO.25.6.066004

2. Daneshjou et al. (2022). Disparities in dermatology AI: assessments 
   using diverse clinical images. *Science Advances*.
   arXiv: 2111.08006

3. Anderson & Parrish (1981). The optics of human skin.
   *J. Invest. Dermatol.* 77(1):13–19.

4. Ferreira et al. (2019). Automatic focus assessment on dermoscopic 
   images acquired with smartphones. *Sensors* 19(22).
   DOI: 10.3390/s19224957

5. Wang et al. (2004). Image quality assessment: from error visibility 
   to structural similarity. *IEEE Trans. Image Process.* 13(4):600–612.

---

## Important Notice

This is a research prototype. It is not a medical device and is not 
intended for clinical diagnosis. All image interpretation must be 
performed by qualified professionals.

---

*Contact: [snigu002@fiu.edu]*
