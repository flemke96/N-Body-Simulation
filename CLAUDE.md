# CLAUDE.md

## Projekt

Gravitations-Simulation (N-Körper-Problem), aktuell als Dreikörperproblem umgesetzt. Lern-/Hobbyprojekt, kein Paket, keine Tests, kein CI. Überblick für Menschen: `README.md`.

## Struktur

- `3body.ipynb` – gesamter Code (einzige Quelle). Zellen in Reihenfolge: Imports → Anfangsbedingungen → Berechnung → Plot → Animation → GIF speichern.
- `basic_animation.gif` – wird von der letzten Notebook-Zelle überschrieben. Nicht von Hand ändern.

## Code-Konventionen

- Zustand in NumPy-Arrays der Form `[Körper, Zeitschritt, xyz]`: `r` (Position), `v` (Geschwindigkeit), `f` (Kraft), `a` (Beschleunigung); `m` hat die Form `[Körper]`.
- SI-Einheiten, `G` aus `scipy.constants`. Bewegung in der x-y-Ebene (z = 0).
- Globale Parameter: `N`, `Time`, `dt`, `nsteps`, `EPS` (Softening, also Glättung bei sehr kleinen Abständen).
- Kommentare und Bezeichner im Code auf Englisch. README auf Deutsch.

## Bekannte Schwachstellen (bewusst noch nicht behoben)

- Die Animationszelle ist fest auf 3 Körper ausgelegt (`line1`–`line3`).
- Integrator: einfaches explizites Verfahren (Euler-ähnlich), nicht energieerhaltend.
- Softening: `|r|³ + EPS²` statt des üblichen `(|r|² + EPS²)^1.5`.
- Kraftberechnung in reinen Python-Schleifen; Aufwand O(N² · nsteps).
- Ungenutzt: `normal`, `Enum`, `clear_output`, `size`, `v0_max` (Reste einer geplanten `Body`-Klasse bzw. von Zufallsstartwerten).

Diese Punkte nur ändern, wenn der Nutzer darum bittet.

## Arbeiten mit dem Notebook

- Programmatisch bearbeiten: als JSON lesen/schreiben (`nbformat` oder `json`) oder mit dem NotebookEdit-Tool. Outputs nicht unnötig verändern.
- Ausführen ohne GUI: `jupyter nbconvert --to notebook --execute 3body.ipynb` (dauert etwa 1 Minute; überschreibt dabei das GIF).
- Abhängigkeiten: `numpy scipy matplotlib pillow` (ffmpeg optional).
