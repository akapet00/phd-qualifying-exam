# PhD Qualifying Exam

This repository contains the code I used for generating figures and performing analysis that pertains to my PhD qualifying exam.

To reproduce the results, easiest way is to create a local environment by using `conda` as
```shell
conda create --name phd-qualifying-exam python=3.9.12
```
and, inside the environment, within `code` repository, run the following command
```shell
pip install -r requirements.txt
```
to install all dependencies listed in `requirements.txt`.

## Contents

| Directory | Subdirectory/Contents | Description |
|:---:|:---:|:---:|
| `LaTeX` |  | All tex files and the list of references used for writing of the PhD qualifying exam. |
| `artwork` |  | All visuals used for writing the PhD qualifying exam. The rest of the figures not listed below are generated by using Python scripts of the same name in `code` directory. **Note**: Two figures, Diao2020Assessment_figure2.png and Kapetanovic2022Assessment_figure2.pdf, have been taken from Diao et al 2020 Phys. Med. Biol. 65 224001 and Kapetanović et al 2022 IEEE J. Electromagn. radiofrequency Microw. Med. Early Access, respectively. |
| 1 | averaging_surface.drawio | Drawio file used for creating averaging_surface.a.pdf and averaging_surface.b.pdf figures which depict ontrol surfaces for averaging on the multi-layer tissue-equivalent block model from 3-D point of view and later point of view, respectively. |
| 2 | em_spectrum.tex | Tikz file for the diagram of electromagnetic spectrum distribution as a function of wavelengths. |
| 3 | em_waves.tex | Tikz file for the visual representation of a propagating plane electromagnetic wave in classical electrodynamics. |
| 4 | exposed_tissue_volume.drawio | Drawio file used for creating exposed_tissue_volume.pdf which depicts exposed 10-g cubic volume for the assessment of the local exposure to electromagnetic fields. |
| `code` |  | Python files used for generating most of the figures within `artwork` directory. Each Python file is used for a single figure of the same name. |
| 1 | cylindrical_model.py | Cylindrical model with radius set to 5 cm. |
| 2 | cylindrical_surface.py | Cylindrical averaging surface of 4 squared centimeters on the surface of the cylindrical model. |
| 3 | pd_decay.py | Power density as a function of frequency. |
| 4 | penetration_depth.py | Power transmission coefficient and power penetration depth into dry skin as functions of frequency. |
| 5 | reference_levels.py | Reference levels for occupational exposure and general public averaged over 6-min interval at the 6-300 range. |
| 6 | research_compilation_2021.py | Papers published in 2021 related to research on bioeffects of radiofrequency fields and/or mobile communications Compiled from the database of publications at [emf-portal](emf-portal.org). |
| 7 | research_compilation.py | Papers published over years from 1950 to 2021 related to research on bioeffects of radiofrequency fields and/or mobile communications. Compiled from the database of publications at [emf-portal](emf-portal.org). |
| 8 | sar_decay.py | Specific absorption rate as a function of depth into the homogeneous block of tissue with dielectric properties of dry skin. |
| 9 | spherical_model.py | Spherical model with radius set to 5 cm. |
| 10 | spherical_surface.py | Spherical averaging surface of 4 squared centimeters on the surface of the spherical model. |
| 11 | utils.py | Auxiliary functions module. |
| `data` |  | Datasets storage. |
| 1 | paper_count.ods | The dataset containg the number of papers by year, from 1950 onward, classified as experimental, epidemiological, dosimetric/technical or review studies related to the bioeffects of electromagnetic fields. **Note**: In order to be able to read the .ods file which is used for storing the data with `pandas`, `odfpy` should be installed. |
| 2 | tissue_properties.csv | The dataset containg the frequency, conductivity, relative permittivity, loss tangetn and penetration depth for different tissues at the 1-100 GHz frequency range. |
| `defense` |  | Presentation files and the presentation for the defense of the PhD qualifying exam. |
| `docs` |  | Documentation related to the application for the PhD qualifying exam. |


 ## License

 [MIT](https://en.wikipedia.org/wiki/MIT_License) except for Diao2020Assessment_figure2.png and Kapetanovic2022Assessment_figure2.pdf files within `artwork` directory which are protected under [CC-BY](https://en.wikipedia.org/wiki/Creative_Commons_license) license protection.