# 2025ESWContest_free_AIM 수정필요!!!!

# Enhanced Ultrasonic Directional Speaker
Original Repository: [gururise/directional_speaker](https://github.com/gururise/directional_speaker)
Original REPO Youtube link: [YouTube](https://www.youtube.com/watch?v=9hD5FPVSsV0)

The sound quality of my directional speaker has improved significantly after adding a decoupling capacitor to the TC4427 MOSFET driver.
Additionally, I implemented an inductor-based low-pass filter (LPF).
As a result, the circuit now includes both an active high-pass filter (HPF) and a passive LPF using an inductor. The LPF uses the transducer's capacitance as part of the filter, meaning the inductance value must be adjusted dynamically depending on the number of transducers used.

## Repository Contents
1. C++ Code for STM32F103C8T6 Microcontroller

There are two versions of the firmware:

One for the Arduino IDE
One for STM32CubeIDE

⚠️ Depending on the board you're using, you need to check which pin is assigned to the onboard LED. Either modify the code accordingly or use the correct version that matches your board's configuration.

2. KiCad Schematic

The full schematic of the directional speaker circuit is provided in KiCad format.

## Board Information
This project uses the STM32F103C8T6 WeAct BluePill Plus board:
🔗 [WeActStudio/BluePill-Plus GitHub](https://github.com/WeActStudio/BluePill-Plus)

## Transducer
I used 42 units of the V40AN16T transducer, which operates at 40 kHz.
Make sure to use transducers that match this frequency.

## Can I Use Other Op-Amps or MOSFET Drivers?
Yes! You're free to upgrade the components.

For example, you can use a MOSFET driver + discrete MOSFET configuration for better performance.
As you might know, using only a MOSFET driver to amplify the PWM signal isn’t ideal, but it still works for this application.

---

## Licensing

This project is a composite work, and its use is governed by the licenses of its constituent components. The project as a whole is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, due to its dependency on Ultralytics YOLOv8.

### Component Licenses

The software components used in this project are licensed as follows:

| Component                               | License                                                                         | Type                  | Key Obligations                                     |
| --------------------------------------- | ------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------- |
| **Ultralytics YOLOv8**                  | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html)                            | Strong Copyleft       | **Source code disclosure required for distribution**|
| **TensorFlow / Keras**                  | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                 | Permissive            | Retain copyright/license notices                    |
| **OpenCV**                              | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)                 | Permissive            | Retain copyright/license notices                    |
| **NumPy / scikit-learn**                | [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause)              | Permissive            | Retain copyright/license notices                    |
| **Pygame**                              | [LGPL 2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)               | Weak Copyleft         | Disclose modifications to the library itself        |

### Summary

-   **For Personal/Academic Use**: You are free to use, modify, and distribute this project for non-commercial, academic, or personal purposes, provided you comply with the terms of the AGPL-3.0 license (which primarily involves making your source code available).
-   **For Commercial Use**: If you wish to use this project in a commercial product without being subject to the source code disclosure requirements of the AGPL-3.0, you **must obtain a separate commercial license for YOLOv8 from Ultralytics**. The other components (TensorFlow, OpenCV, etc.) are commercially friendly.

For the full text of the governing license for this project, please see the [LICENSE](LICENSE) file.
