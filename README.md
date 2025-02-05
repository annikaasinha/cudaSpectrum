

#### Project Title
**FFTAccelerator: High-Performance FFT Computations on GPUs**

#### Project Description
FFTAccelerator is designed to showcase the powerful capabilities of NVIDIA's CUDA technology, specifically utilizing the cuFFT library to perform fast Fourier transforms (FFT). This project demonstrates the acceleration of FFT computations on GPUs, aiming to provide a tool for efficient spectral analysis in scientific and engineering applications.

#### Objectives
- To implement FFT computations using the CUDA cuFFT library.
- To demonstrate the performance benefits of using GPU acceleration for FFT operations.
- To provide a flexible command-line interface for easy manipulation of FFT computations on various data sizes.

#### Installation
##### Prerequisites:
- NVIDIA GPU with CUDA Compute Capability 5.0 or higher.
- CUDA Toolkit 11.0 or later.
- Modern C++ compiler compatible with C++14 standard.

##### Build Instructions:
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FFTAccelerator.git
   ```
2. Navigate to the project directory:
   ```bash
   cd FFTAccelerator
   ```
3. Build the project using the Makefile:
   ```bash
   make
   ```

#### Usage
To run FFTAccelerator, execute the following command from the project root directory:
```bash
./build/fftExample -n=<number_of_elements>
```
Replace `<number_of_elements>` with the desired size of the data set for FFT computation.

#### Technologies Used
- **CUDA/C++**: Utilized for GPU-accelerated FFT computations.
- **cuFFT Library**: NVIDIA's library for efficient FFT operations on CUDA-enabled GPUs.

#### Contributing
Contributions to FFTAccelerator are welcome. Please fork the repository, make your changes, and submit a pull request for review.

#### Data Sources
This project does not require any external data sources but is designed to work with synthetic data generated within the application for demonstration purposes.

#### Authors
- Annika Sinha(https://github.com/annikaasinha) - Initial development and design.

 [FFTAccelerator Demonstration Video](https://youtube.com)

#### Proof of Execution
![image](https://github.com/user-attachments/assets/9a7eb0ac-3b74-4e7d-8a87-cc275c0ee2b0)
