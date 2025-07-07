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

