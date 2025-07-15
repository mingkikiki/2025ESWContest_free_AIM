# 2025ESWContest_free_AIM 

## 인식된 객체를 위한 초지향성 사운드 타겟팅
획일적인 정보 제공 방식에서 벗어나, 관람객의 특성에 맞춰 설명의 난이도를 조절하고 집중도를 높여주려는 아이디어에서 출발했습니다. 이는 관람객 개개인에게 더욱 풍부하고 몰입감 있는 문화 경험을 제공하고자 하는 목표에서 비롯되었습니다.

도슨트 서비스 현장에서 특정 대상에게만 정보를 전달해야 하는 필요성을 느끼고, 초음파의 비선형성을 이용해 공기 중에서 소리를 복조시키는 초지향성 스피커 기술이 이 문제에 대한 혁신적인 해결책이 될 수 있다고 판단했습니다. 특히, 소리가 전파되는 가청 영역을 제어하여 불필요한 소음 없이 원하는 대상에게만 정보를 전달할 수 있다는 점에 매료되었습니다.

따라서 성인과 아이를 실시간으로 인식하고, 인식된 객체의 특성에 맞춰 난이도가 조절된 음성 해설을 초지향성 스피커를 통해 해당 객체에게만 선별적으로 전송하는 맞춤형 도슨트 시스템을 개발하였습니다.

## 코드 정보
**src/STM32**

PWM 작업을 수행하는 코드 입니다.PB2와 PC13 두가지 버전이 있습니다. 사용한 보드의 내장 LED 핀 번호에 맞는 코드를 사용하면 됩니다.
또한 .ioc 파일을 이용하여 STM32CUBEIDE에서 프로젝트 로드를 할 수 있습니다. 

**/src/STM32/Circuit Diagram**

회로도가 포함되어 있습니다. KiCad로 작성되었습니다.

**/src/AI**

객체인식 시스템의 파이썬 코드가 있습니다.

## MCU 보드 정보
STM32F103C8T6을 사용하였습니다.
🔗 [WeActStudio/BluePill-Plus GitHub](https://github.com/WeActStudio/BluePill-Plus)

---

## Licensing

This project is a composite work, and its use is governed by the licenses of its constituent components. The project as a whole is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**, due to its dependency on Ultralytics YOLOv8.

### Component Licenses

The software components used in this project are licensed as follows:

| Component                               | License                                                                         | Type                  | Key Obligations                                     |
| --------------------------------------- | ------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------- |
| **Ultralytics YOLOv8**                  | AGPL-3.0                            | Strong Copyleft       | **Source code disclosure required for distribution**|
| **TensorFlow / Keras**                  | Apache License 2.0                | Permissive            | Retain copyright/license notices                    |
| **OpenCV**                              | Apache License 2.0                | Permissive            | Retain copyright/license notices                    |
| **NumPy / scikit-learn**                | BSD 3-Clause License              | Permissive            | Retain copyright/license notices                    |
| **Pygame**                              | LGPL 2.1               | Weak Copyleft         | Disclose modifications to the library itself        |
| **STM32 PWM code**                      | GPL 2.0       | Strong Copyleft       | Source code disclosure required when distributing   |

### Summary

-   **For Personal/Academic Use**: You are free to use, modify, and distribute this project for non-commercial, academic, or personal purposes, provided you comply with the terms of the AGPL-3.0 **and GPL-2.0** licenses (which primarily involve making your source code available).
-   **For Commercial Use**: If you wish to use this project in a commercial product without being subject to the source code disclosure requirements of the AGPL-3.0 or GPL-2.0, you **must obtain a separate commercial license for YOLOv8 from Ultralytics**, and ensure compatibility or relicensing for any GPL-2.0 components. The other components (TensorFlow, OpenCV, etc.) are commercially friendly.

-   **Note**: The STM32 PWM code is fully independent from the AI-related components (e.g., YOLOv8, TensorFlow, etc.), and is included in the repository for use in hardware control scenarios.


For the full text of the governing license for this project, please see the [LICENSE](LICENSE) file.
