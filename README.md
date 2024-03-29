# Gpixel-GSPRINT4521-Characterization
> #### *21MP High Speed CMOS Image Sensor.*

## EMVA Standard 1288

The different parameters that describe the characteristics and quality of a sensor are gathered and coherently described in the [EMVA 1288](https://www.emva.org/standards-technology/emva-1288/) (Release 3.1).

I would highly recommend that you carefully read and understand this document before starting any operation.

This standard illustrates the fundamental parameters that must be given to fully describe the real behavior of a sensor, together with the well-defined measurement methods to get these parameters. 

The standard parameters are:
- Dark current (DC) [ADU/s]
- Quantum efficiency (QE) [%]
- Read noise (RON) [e-]
- Gain (K) [ADU/e-]
- Signal-to-noise ratio (SNR) [dB]
- Dynamic range (DR) [dB]
- Saturation (full-well) capacity [e-] 
- Photo-Response Non-Uniformity (PRNU) [%]
- Dark Signal Non-Uniformity (DSNU) [e-] 

For the time being, QE, PRNU and DSNU will not be discussed further here. 

![Plots](https://github.com/NHL-B/Gpixel-GSPRINT4521-Characterization/blob/main/doc/images/plots.png)

[Requirements](requirements.txt): A list of Python libraries you'll need for this project.

## To use
- Fork this repo
- Clone to disk in a stable path, such as C or your user folder
- Edit the scripts to your liking
- If you find any problems or want to suggest modifications, create an issue.

## Additional links

[EMVA](https://www.emva.org/ "EMVA - European Machine Vision Association") • [Camera Test Protocol w/ ImageJ](https://www.photometrics.com/wp-content/uploads/2019/10/Technical-Notes-Camera-Test-Protocol-November-2019.pdf "Camera Test Protocol") • [Gpixel GSPRINT4521](https://www.gpixel.com/products/area-scan-en/sprint/gsprint4521/ "Gpixel - GSPRINT4521")

## License & copyright
© NHL-B
Licensed under the [MIT License](LICENSE).
