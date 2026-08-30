# UAV-VLRR: Vision-Language Informed NMPC for Rapid Response in UAV Search and Rescue

Yasheerah Yaqoot<sup>∗</sup> , Muhammad Ahsan Mustafa<sup>∗</sup> , Oleg Sautenkov, Artem Lykov, Valerii Serpiva, and Dzmitry Tsetserukou

*Abstract*— Emergency search and rescue (SAR) operations often require rapid and precise target identification in complex environments where traditional manual drone control is inefficient. In order to address these scenarios, a rapid SAR system, UAV-VLRR (Vision-Language-Rapid-Response), is developed in this research. This system consists of two aspects: 1) A multimodal system which harnesses the power of Visual Language Model (VLM) and the natural language processing capabilities of ChatGPT-4o (LLM) for scene interpretation. 2) A non-linear model predictive control (NMPC) with builtin obstacle avoidance for rapid response by a drone to fly according to the output of the multimodal system. This work aims at improving response times in emergency SAR operations by providing a more intuitive and natural approach to the operator to plan the SAR mission while allowing the drone to carry out that mission in a rapid and safe manner. When tested, our approach was faster on an average by 33.75% when compared with an off-the-shelf autopilot and 54.6% when compared with a human pilot.

Github:<https://github.com/ahsan-mustafa/uav-vlrr> Video of UAV-VLRR:<https://youtu.be/KJqQGKKt1xY> *Keywords: VLM; LLM-agents; VLM-agents; UAV; Navigation; Drone; Path Planning; NMPC.*

## I. INTRODUCTION

Search and rescue (SAR) operations in disaster-stricken environments require fast and efficient situational assessment to locate survivors and critical infrastructure. Unmanned Aerial Vehicles (UAVs) have become vital in SAR missions due to their ability to access hard-to-reach areas, provide real-time imagery, and reduce response times [1], [2]. However, traditional UAV-based SAR relies heavily on manual flight control or waypoint setting. In high-stakes emergencies, the pressure can overwhelm even experienced responders, and the chaotic nature of disaster zones often leads to impaired judgment and delays in mission planning. As cognitive overload increases, critical details may be overlooked, and manual approaches can falter, as seen in our previous work FlightAR [3]. These limitations highlight the need for an intelligent SAR system that can autonomously generate mission waypoints in complex environments with minimal human input. Furthermore, such a system must be deployed on a UAV capable of executing missions safely and rapidly.

The authors are with the Intelligent Space Robotics Laboratory, Center for Digital Engineering, Skolkovo Institute of Science and Technology. {yasheerah.yaqoot, ahsan.mustafa, oleg.sautenkov, artem.lykov, valerii.serpiva, d.tsetserukou}@skoltech.ru

![](_page_0_Figure_9.jpeg)

Fig. 1: Illustration of the UAV-VLRR framework. The left image shows the input to the system, and the right displays the identified points by the multimodal system. Below, the NMPC guides the drone's trajectory, ensuring obstacle avoidance and navigation to target points.

A key challenge in achieving autonomous UAV-based SAR missions lies in environmental perception and realtime decision-making. Traditional UAV mission planning techniques often depend on handcrafted obstacle maps, LiDAR-based navigation, or heuristic path-planning algorithms. While effective in structured environments, these approaches struggle to adapt to the unpredictable nature of disaster zones where obstacles, such as collapsed buildings, debris, and vegetation, are constantly changing. To address these limitations, there is a need for a system that can autonomously interpret aerial imagery, extract relevant information, and generate actionable flight paths in real time. One of our previous research [4] involves a UAV-VLA framework built on this concept.

In this work, we build on the UAV-VLA framework [4] by integrating its capability of interpreting aerial images with

<sup>\*</sup>These authors contributed equally to this work.

the agile control of a quadrotor resulting in quick coverage of the destination points given by the multimodal system. Our contributions are as follows:

- We introduce the UAV-VLRR framework, combining the multimodal Vision-Language interpretation of aerial images with rapid control.
- We apply a point-to-point Non-linear Model Predictive Control (NMPC) control scheme with built in obstacle avoidance to ensure safe and rapid UAV response in complex environments.
- We demonstrate that our framework outperforms other traditional approaches in the field of drone search and rescue.

## II. RELATED WORK

## *A. Multimodal Vision-Language Approaches for Robotic Systems*

The introduction of Vision Transformers (ViTs) [5], [6] marked a pivotal shift in the development of models capable of integrating various input and output modalities, including text, images, and video. This progress laid the foundation for models such as OpenAI's ChatGPT-4 Omni [7], which can perform real-time reasoning across multiple modalities, enhancing multimodal interactions. In the robotics domain, the Allen Institute for AI introduced the Molmo model, which uses image-text pairs to locate objects in response to user requests [8], further advancing the integration of vision and language in robotic systems.

Vision-Language models have also been applied in UAV control. Sautenkov et al. [3] improved drone surveillance using multiple video streams and object detection, although manual operation was still required. The Google DeepMind's RT-2 [9] introduced models advanced this field by enabling direct robot control from multimodal sensory inputs. The UAV-VLA framework, as presented in [4], takes these advancements further by using multimodal systems to generate actionable mission paths through text-image pairs. This approach underscores the critical role of vision and language integration in a variety of robotic applications, particularly in tasks that require real-time environmental understanding.

Further expanding this line of research, UAV-CodeAgents [10] introduced reasoning step for navigation, and UAV-VLPA\* [11] introduced global route optimization by combining TSP and A\* path planning, significantly reducing trajectory lengths in large-scale UAV missions. RaceVLA [12] and CognitiveDrone [13] applied VLA models to racing drones and drone reasoning, producing real-time velocity and yaw commands from FPV video and language inputs, and achieving human-like decision-making in dynamic racing environments. These advances highlight the growing versatility of vision-language-action systems in aerial robotics.

Building on these developments, UAV-VLRR focuses on real-time mission execution in cluttered environments by integrating semantic understanding with onboard NMPC, enabling fast and safe UAV operation for critical applications such as search and rescue.

### *B. Safe Agile Control for Drones*

The importance of NMPC for agile drones can be seen in its use by the drone racing team at ETH Zurich, which are the best in the world for high-speed drone control. They have used NMPC in many of their works [14], [15], [16], [17]. Sun et al. [14] did a comparative study between NMPC and DFBC in which NMPC outperformed DFBC in terms of tracking dynamically infeasible trajectories, although it required significantly higher computational resources, which could be a bottleneck in real-time systems. This study provided critical insights into the trade-offs between computational efficiency and control performance. Romero et al. [15] tackled the agile drone problem by using a model predictive contouring control approach that resulted in time-optimal trajectories in real-time with effective high-speed control. However, their work was computationally expensive and they have stated that the controller was not run onboard the drone but rather on an external computer. Hanover et al. [16] used an adaptive MPC approach by cascading the MPC with an L1-Adaptive Controller. This resulted in immediate model mismatches and disturbances very effectively, but they have stated that there is potential for violating actuator constraints due to the inner cascaded loop. Torrente et al. [17] used a data-driven mpc approach by modeling aerodynamic forces using Gaussian processes, but the controller was run offboard here as well. In [18], Ramezani et al. implemented obstacle avoidance in their MPC framework along with using long-short-term memory for states predicition. However, this work was performed only in simulation while using a simplified 3-DOF kinematics drone model. Moreover, one of our previous works, SafeSwarm [19], worked on safe drone landings in crowded areas, which is an important factor in crowded emergency scenarios.

In order for the drone to fly at high speeds in a satisfactory manner with a minimal dynamics model (without any compensation for drag or aerodynamic mismatches), a pointto-point NMPC technique is utilized in this paper, unlike most of the work mentioned in the literature, which focus on first an external trajectory generation module and then a trajectory following module. This approach enables the drone to fly properly despite the model mismatches since now it does not have strict constraints of tracking a given trajectory. Moreover, this also simplifies the computational need and hence is able to be deployed on an onboard computer like an OrangePi in our case. This approach is also advantageous in the sense that it only requires the target points and the obstacle points, which perfectly fits in the pipeline when cascading with the multimodal system.

## III. SYSTEM OVERVIEW

## *A. Vision-Language Integration for Accurate Object Identification*

In this work, a multimodal system comprising a Large Language Model (LLM) and a Vision-Language Model (VLM) is employed to enhance environmental understanding, as illustrated in Fig. [2.](#page-2-0) ChatGPT-4o serves as the LLM

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Fig. 2: System architecture of the UAV-VLRR framework.

agent, responsible for extracting goal objects, specifically "target points" and "obstacles". The quantized Molmo-7B-D BnB 4-bit model [20] is utilized as the VLM agent for image processing and goal object identification.

The system processes an image-text pair as input, which is handled by both the LLM and VLM agents. The image-text pair processing can be mathematically represented as:

$$C = f_{\text{LIM VIM}}(I, T), \tag{1}$$

where I is the input image and T is the input text, and  $\mathcal{C}$  represents the output coordinates of the identified goal objects.

Once the goal objects are identified, their pixel coordinates are mapped onto the image and converted into real-world coordinates using image metadata. Specifically, the real-world coordinates are computed based on the camera's height and field-of-view (FoV) parameters. The horizontal and vertical real-world dimensions in meters are first calculated from the diagonal FoV and camera height using the following formulas:

Real width (m) = 
$$2 \cdot h_{\text{camera}} \cdot \tan\left(\frac{\theta_{\text{horizontal}}}{2}\right)$$
, (2)

Real height (m) = 
$$2 \cdot h_{\text{camera}} \cdot \tan\left(\frac{\theta_{\text{vertical}}}{2}\right)$$
, (3)

where  $h_{\rm camera}$  is the camera's height above the ground, and  $\theta_{\rm horizontal}$  and  $\theta_{\rm vertical}$  are the horizontal and vertical FoVs, respectively, which are derived from the diagonal FoV and aspect ratio of the camera.

Once the real-world dimensions of the image are known, they are used to define the Cartesian coordinate bounds for the image. The normalized pixel coordinates of detected objects are mapped into real-world Cartesian coordinates by scaling them according to the image's real-world dimensions. These real-world coordinates represent the target points and obstacles, which are then passed to the NMPC for task execution.

#### B. NMPC for Rapid Response

The non-linear model predictive control in this research follows a point-to-point architecture. In addition, the objective function has a penalty term associated with the obstacle points received from the multimodal system. This NMPC setup enables the controller to not depend on any external trajectory generation technique. The NMPC finds the optimal trajectory and the set of control inputs designed for rapid control.

The dynamics of the quadrotor system is governed by 13 states where  $p_W = \left[p_x, p_y, p_z\right]^T$  are the position coordinates in the world frame,  $v_W = \left[v_x, v_y, v_z\right]^T$  are the linear velocity components in the world frame,  $q_B = \left(q_\omega, q_x, q_y, q_z\right)^T$  are the quaternions for the orientation of the drone's body, and finally  $\omega_B = \left[\omega_x, \omega_y, \omega_z\right]^T$  are the body angular rates.

$$\dot{x} = \begin{bmatrix} \dot{p}_W \\ \dot{v}_W \\ \dot{q}_B \\ \dot{\omega}_B \end{bmatrix} = \begin{bmatrix} v_W \\ R(q)\frac{T_B}{m} + g \\ \frac{1}{2}q_{\omega B} \cdot q_B \\ J^{-1} \left(\tau_B - \omega_B \times J\omega_B\right) \end{bmatrix},$$

where R(q) is the quaternion rotational matrix,  $T_B$ 

 $\left[0,0,\sum_{i=1}^{4}T_{i}\right]^{T}$  is the total thrust, g is the gravitational vector  $g=\left[0,0,9.81\right]^{T}$ ,  $J=\mathrm{diag}(J_{x},J_{y},J_{z})$  is the diagonal of the inertia matrix and  $q_{\omega}=\left(0,\omega_{x},\omega_{y},\omega_{z}\right)^{T}$  is the angular velocity quaternion.

The drone's body torque matrix is according to the free body diagram shown in Fig. 3. The body torque matrix comes out to be:

$$\tau_B = \begin{bmatrix} -l_y & l_y & l_y & -l_y \\ -l_x & -l_x & l_x & l_x \\ k_t & -k_t & k_t & -k_t \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \\ u_3 \\ u_4 \end{bmatrix},$$

where  $u_1, u_2, u_3, u_4$  are the input motor forces,  $l_x, l_y$  are the distances to the x-axis and y-axis, respectively and,  $k_t$  is the torque constant.

<span id="page-3-0"></span>![](_page_3_Picture_4.jpeg)

Fig. 3: Drone free-body diagram.

In order to form a discretized nonlinear optimal control problem, the Runge-Kutta method of 4th order was used:

$$x(k+1) = f_{RK4}(x(k), u(k), \delta t)$$

The NMPC was formulated in a multiple shooting scheme. The constructed optimization problem is as below:

$$l(x, u) = ||x_u - x_r||_Q^2 + ||u||_R^2 + Penalty_{Obs},$$
 (4)

$$\min_{u} J(x, u) = \sum_{k=0}^{N-1} l(x_u(k), u(k)), \tag{5}$$

subject to:

$$\begin{split} x(k+1) &= f_{\text{RK4}}(x(k), u(k), \delta t), \\ x_u(0) &= x_0, \\ u_{min} \leq u(k) \leq u_{max}, \quad \forall k \in [0, N-1], \\ x(k) \in X, \quad \forall k \in [0, N] \\ Obstacle_{x,y} \end{split}$$

The system was discretized into a prediction horizon of N steps with a step horizon of T between each step. The control problem is iteratively solved in real time onboard the drone using CasADi [21].

#### IV. EXPERIMENTAL SETUP

The UAV-VLRR framework was tested inside the drone arena of the Intelligent Space Robotics Lab at Skoltech. The command given to the system was: "Fly around each of the center of the X on yellow objects. Avoid three legs of red tripod stands." The system was tested under various conditions, with the following three experiments performed:

- Exp 1: The drone flies to the target points using the UAV-VLRR framework.
- Exp 2: The drone flies to the target points using an off-the-shelf autopilot.
- Exp 3: A human drone pilot is shown the picture and then flies around the target points while having access to a belly-mounted camera on the drone.

There were two different scenarios in which all the three experiments were conducted:

- Scene 1: There were three target points (X marked on yellow objects) and two obstacles (red tripod stands).
- Scene 2: There were four target points (X marked on yellow objects) and three obstacles (red tripod stands).

The multimodal system ran on a remote server with an RTX 4090 GPU (24GB VRAM) and an Intel Core i9-13900K processor. During the experiment, the drone sent the image and command to the server, which returned the target and obstacle coordinates. The same 2D aerial image and prompt were also given to the human pilot for comparison. Images used in Scene 1 and Scene 2 are shown in Fig. 4.

![](_page_3_Picture_23.jpeg)

<span id="page-3-1"></span>![](_page_3_Picture_24.jpeg)

Fig. 4: Scenes used in the experiment with target points (X on yellow objects) and obstacles (red tripod stands).

Each experiment was timed to assess the speed of execution. For the UAV-VLRR framework, the timing started as soon as the code was launched, while for the human pilot, the timing started as soon as the image was shown. The goal of all experiments was to have the drone fly to the required points and avoid obstacles in the shortest time possible.

#### V. EXPERIMENTAL RESULTS

#### A. Multimodal System Results

The results obtained from the multimodal system were compared to the Vicon data for the target points and obstacles in both scenes. The identified images for Scene 1 and Scene 2 can be seen in Fig. 5, which illustrates the target points and obstacles detected by the system.

Tables I and II present the ground truth values alongside the multimodal system's detected points, as well as the corresponding accuracy for Scene 1 and Scene 2, respectively. For

<span id="page-4-0"></span>![](_page_4_Figure_0.jpeg)

![](_page_4_Figure_1.jpeg)

- (a) Results for Scene 1. (b) Results for Scene 2.

Fig. 5: Identified target points and obstacles from the multimodal system for the given image-text pairs in both scenes.

this analysis, an identification was considered accurate if the detected point was within a 25 cm radius of the actual object. This threshold accounts for the safety radius and obstacle gain applied in the NMPC to ensure safety during navigation.

It is worth noting that the image was not captured from a very high altitude, which may have resulted in some distortion or skewing. As a result, there was a higher error in the identification of some of the objects, but this is expected due to the imaging conditions at the time of capture.

<span id="page-4-1"></span>TABLE I. COMPARISON OF VICON (GROUND TRUTH) AND MUL-TIMODAL SYSTEM FOR SCENE 1 WITH ERROR VALUES

| Scene 1    |   |             |             |       |  |  |
|------------|---|-------------|-------------|-------|--|--|
|            |   | Vicon       | Multimodal  | Error |  |  |
|            |   | coordinates | coordinates | (cm)  |  |  |
| Target 1   | X | -1.42       | -1.27       | 15    |  |  |
|            | Y | -1.39       | -1.25       | 14    |  |  |
| Target 2   | X | 1.43        | 1.34        | 9     |  |  |
|            | Y | 0.13        | 0.18        | 5     |  |  |
| Target 3   | X | -1.76       | -1.60       | 16    |  |  |
|            | Y | 1.82        | 1.73        | 9     |  |  |
| Obstacle 1 | X | -0.41       | -0.28       | 13    |  |  |
|            | Y | -0.72       | -0.65       | 7     |  |  |
| Obstacle 2 | X | -0.85       | -0.66       | 19    |  |  |
|            | Y | 1.31        | 1.20        | 11    |  |  |

### *B. Mission Results*

The times for each of the experiments are listed in Table [III.](#page-4-3) For scene 1, it can be observed that experiment

<span id="page-4-2"></span>TABLE II. COMPARISON OF VICON (GROUND TRUTH) AND MULTIMODAL SYSTEM FOR SCENE 2 WITH ERROR VALUES

| Scene 2    |   |             |             |       |  |  |  |
|------------|---|-------------|-------------|-------|--|--|--|
|            |   | Vicon       | Multimodal  | Error |  |  |  |
|            |   | coordinates | coordinates | (cm)  |  |  |  |
| Target 1   | X | -1.42       | -1.42       | 0     |  |  |  |
|            | Y | -1.39       | -1.38       | 1     |  |  |  |
| Target 2   | X | 1.56        | 1.44        | 12    |  |  |  |
|            | Y | -1.43       | -1.25       | 18    |  |  |  |
| Target 3   | X | 1.73        | 1.50        | 23    |  |  |  |
|            | Y | 1.70        | 1.60        | 10    |  |  |  |
| Target 4   | X | -1.76       | -1.55       | 21    |  |  |  |
|            | Y | 1.82        | 1.70        | 12    |  |  |  |
| Obstacle 1 | X | -1.29       | -1.16       | 13    |  |  |  |
|            | Y | 0.28        | 0.26        | 2     |  |  |  |
| Obstacle 2 | X | 0.08        | 0.14        | 6     |  |  |  |
|            | Y | 1.57        | 1.38        | 19    |  |  |  |
| Obstacle 3 | X | -0.15       | -0.03       | 12    |  |  |  |
|            | Y | -1.35       | -1.25       | 10    |  |  |  |

1 achieved the fastest time to complete the mission in scene 1 with 28 seconds, while experiment 3 took the most time to complete the mission with 57 seconds. Experiment 1 was 30% faster than experiment 2 and 50.9% faster than experiment 3. In scene 2, once again experiment 1 achieved the fastest time for mission completion with 30 seconds while experiment 3 was again the slowest with 72 seconds. in this scene, Experiment 1 was 37.5% faster than experiment 2 and 58.3% faster than experiment 3. It can be deduced from the experiments that the multimodal setup (Experiments 1 and 2) outperformed the human pilots. Moreover, during manual flights, the human pilot was more prone to crashing into the obstacles. When comparing the flight times of experiment 1 and 2, it was evident that the custom NMPC was able to complete the mission faster than the off-the-shelf autopilot and was consistently better.

TABLE III. FLIGHT RESULTS

<span id="page-4-3"></span>

|                            | Exp 1 | Exp 2 | Exp 3 |
|----------------------------|-------|-------|-------|
| Time Taken for Scene 1 (s) | 28    | 40    | 57    |
| Time Taken for Scene 2 (s) | 30    | 48    | 72    |

The flight trajectories for scene 1 and 2 experiments are shown in Fig. [6](#page-4-4) and Fig. [7](#page-5-0) respectively.

<span id="page-4-4"></span>![](_page_4_Figure_17.jpeg)

- (a) Scene 1 UAV-VLRR. (b) Scene 1 Off-the-Shelf Autopilot.

![](_page_4_Figure_20.jpeg)

(c) Scene 1 - Human Pilot.

Fig. 6: Flight paths for the 3 experiments for Scene 1.

## VI. CONCLUSION

In this work, we present the UAV-VLRR framework which is aimed at improving emergency response times in drone search and rescue operations. We demonstrate that our framework outperforms other traditional approaches in the field of drone search and rescue:

<span id="page-5-0"></span>![](_page_5_Figure_0.jpeg)

(a) Scene 2 - UAV-VLRR. (b) Scene 2 - Off-the-Shelf Autopilot.

![](_page_5_Figure_3.jpeg)

(c) Scene 2 - Human Pilot.

Fig. 7: Flight paths for the 3 experiments for Scene 2.

- The text input provided a more natural way for the operator to design the search and rescue mission rather than observing the image and manually entering in waypoints.
- The point-to-point NMPC provided rapid response for quick mission completion.
- The amalgamation of these 2 aspects resulted in much shorter times for completion of missions.
- Our framework was tested on 2 different scenarios and was faster on an average by 33.75% when compared with off-the-shelf autopilot and 54.6% when compared with a human pilot.

These enhanced response times can be crucial in real-life scenarios where a matter of few seconds can prove to be very important.

## VII. FUTURE WORK

Future work on the UAV-VLRR system will focus on incorporating adaptive learning techniques to improve performance over time. In addition, exploring real-time coordination between multiple UAVs could enhance coverage and efficiency in large-scale SAR operations.

Another important direction is the integration of dynamic environmental factors, such as moving obstacles, to improve the system's relevance and robustness in real-world conditions. This will enable the UAVs to better navigate complex and unpredictable scenarios, which are common in search and rescue missions.

## ACKNOWLEDGEMENTS

Research reported in this publication was financially supported by the RSF grant No. 24-41-02039.

### REFERENCES

- [1] M. Lyu, Y. Zhao, C. Huang, and H. Huang, "Unmanned aerial vehicles for search and rescue: A survey," *Remote Sensing*, vol. 15, no. 13, 2023.
- [2] C. Vincent-Lambert, A. Pretorius, and B. Van Tonder, "Use of unmanned aerial vehicles in wilderness search and rescue operations: A scoping review," *Wilderness & Environmental Medicine*, vol. 34, no. 4, pp. 580–588, 2023.
- [3] O. Sautenkov, S. Asfaw, Y. Yaqoot, M. A. Mustafa, A. Fedoseev, D. Trinitatova, and D. Tsetserukou, "FlightAR: AR flight assistance interface with multiple video streams and object detection aimed at immersive drone control," in *2024 IEEE Int. Conf. on Robotics and Biomimetics (ROBIO)*, 2024, pp. 614–619.
- [4] O. Sautenkov, Y. Yaqoot, A. Lykov, M. A. Mustafa, G. Tadevosyan, A. Akhmetkazy, M. Altamirano Cabrera, M. Martynov, S. Karaf, and D. Tsetserukou, "UAV-VLA: Vision-language-action system for large scale aerial mission generation," in *Proc. of the 2025 ACM/IEEE Int. Conf. on Human-Robot Interaction*, 2025, p. 1588–1592.
- [5] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," 2021, arXiv:2010.11929.
- [6] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever, "Learning transferable visual models from natural language supervision," in *Int. Conf. on Machine Learning*, 2021, p. 8748–8763.
- [7] OpenAI *et al.*, "GPT-4 technical report," 2024, arXiv:2303.08774.
- [8] M. Deitke *et al.*, "Molmo and PixMo: Open weights and open data for state-of-the-art multimodal models," 2024, arXiv:2409.17146.
- [9] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski *et al.*, "RT-2: Vision-language-action models transfer web knowledge to robotic control," 2023, arXiv:2307.15818.
- [10] O. Sautenkov, Y. Yaqoot, M. A. Mustafa, F. Batool, J. Sam, A. Lykov, C.-Y. Wen, and D. Tsetserukou, "Uav-codeagents: Scalable uav mission planning via multi-agent react and vision-language reasoning," 2025. [Online]. Available:<https://arxiv.org/abs/2505.07236>
- [11] O. Sautenkov, A. Akhmetkazy, Y. Yaqoot, M. A. Mustafa, G. Tadevosyan, A. Lykov, and D. Tsetserukou, "UAV-VLPA\*: A visionlanguage-path-action system for optimal route generation on a large scales," 2025, arXiv:2503.02454.
- [12] V. Serpiva, A. Lykov, A. Myshlyaev, M. H. Khan, A. A. Abdulkarim, O. Sautenkov, and D. Tsetserukou, "RaceVLA: VLAbased racing drone navigation with human-like behaviour," 2025, arXiv:2503.02572.
- [13] A. Lykov, V. Serpiva, M. H. Khan, O. Sautenkov, A. Myshlyaev, G. Tadevosyan, Y. Yaqoot, and D. Tsetserukou, "CognitiveDrone: A VLA model and evaluation benchmark for real-time cognitive task solving and reasoning in UAVs," 2025, arXiv:2503.01378.
- [14] S. Sun, A. Romero, P. Foehn, E. Kaufmann, and D. Scaramuzza, "A comparative study of nonlinear MPC and differential-flatness-based control for quadrotor agile flight," *IEEE Transactions on Robotics*, vol. 38, no. 6, pp. 3357–3373, 2022.
- [15] A. Romero, S. Sun, P. Foehn, and D. Scaramuzza, "Model predictive contouring control for time-optimal quadrotor flight," *IEEE Transactions on Robotics*, vol. 38, no. 6, pp. 3340–3356, 2022.
- [16] D. Hanover, P. Foehn, S. Sun, E. Kaufmann, and D. Scaramuzza, "Performance, Precision, and Payloads: Adaptive nonlinear MPC for quadrotors," *IEEE Robotics and Automation Letters*, vol. 7, no. 2, pp. 690–697, 2022.
- [17] G. Torrente, E. Kaufmann, P. Fohn, and D. Scaramuzza, "Data-driven ¨ MPC for quadrotors," *IEEE Robotics and Automation Letters*, vol. 6, no. 2, pp. 3769–3776, 2021.
- [18] M. Ramezani, H. Habibi, J. L. Sanchez-Lopez, and H. Voos, "Uav path planning employing MPC-reinforcement learning method considering collision avoidance," in *2023 Int. Conf. on Unmanned Aircraft Systems (ICUAS)*, 2023, pp. 507–514.
- [19] G. Tadevosyan, M. Osipenko, D. Aschu, A. Fedoseev, V. Serpiva, O. Sautenkov, S. Karaf, and D. Tsetserukou, "Safeswarm: Decentralized safe RL for the swarm of drones landing in dense crowds," in *https://arxiv:2501.07566*, 2025.
- [20] *Molmo-7B-D BnB 4bit quantized 7GB*, 2024. [Online]. Available: <https://huggingface.co/cyan2k/molmo-7B-D-bnb-4bit>
- [21] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, "CasADi – a software framework for nonlinear optimization and optimal control," *Mathematical Programming Computation*, vol. 11, no. 1, pp. 1–36, 2019.