# Water Heater Modeling

A 2D numerical simulation of heat transfer in a water heater tank.

## Overview

This project models the temperature distribution in a stratified tank with a steel bottom layer and water layer, accounting for:
- **Heat conduction** through steel and water
- **Convection** due to water flow
- **Volumetric heat source** from the heating element
- **Thermal properties** of both materials

## Features

- Solves steady-state heat equation using finite difference method
- Sparse matrix assembly with upwind convection scheme
- Visualization of 2D temperature fields and vertical profiles

## Usage

```bash
python main_sol_stat.py
```

Adjust parameters in the main script (geometry, thermal properties, mesh size) to simulate different configurations.

## Output

- `Iso_Temp_I*_J*.png` – 2D temperature field contour plot
- `Profil_T_y_I*_J*.png` – Vertical temperature profiles at different x-positions

## Results

<img width="1524" height="881" alt="image" src="https://github.com/user-attachments/assets/edf7a5c6-8fdb-441d-9d75-cfd1cc4a80bf" />

<img width="1430" height="895" alt="image" src="https://github.com/user-attachments/assets/825b04c1-034e-419e-b740-c10aac12412d" />


## Authors

Nolan C., Marin D.
