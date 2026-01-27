
import numpy as np
import Lib_sol_stat_save as lib
import matplotlib.pyplot as plt

# Travail de Nolan C. et Marin D.

# Données géométriques 
L = 1.0 
Hs = 0.025
Hf = 0.01
Ht = Hs+Hf

# Données physique
ks = 16.0    # Conductivité thermique acier en W m^-1 K^-1  
kf = 0.6     # Conductivité thermique de l'eau
rhof = 200000  # masse volumique de l'eau
cf = 4180    # capacité calorifique massique de l'eau 
Te = 300.0     # température d'entrée de l'eau 
umax = 0.001    # vitesse max de l'eau 
f = 100000.0   # puissance thermique volumique dissipée dans le solide
xl = 0.1*L      # abscisse gauche source chaleur 
xr = 0.9*L      # abscisse droite source chaleur 
yd = -0.8 * Hs  # ordonnée inf. source chaleur 
yu = -0.2 * Hs  # ordonnée sup. source chaleur 

# La boucle c'est pour la question 5.2 
for I, J in zip([50], [50]):
    # Paramètres Numériques
    #I = 81
    #J = 81
    dx = L/(I-1)                        # Pas de maillage en x 
    dy = Ht/(J-1)                       # Pas de maillage en y
    xn = np.linspace(0., L, I)          # Abscisses des noeuds 
    yn = np.linspace(-Hs, Hf, J)        # Ordonnées des noeuds  
    X,Y = np.meshgrid(xn, yn)           # Coordonnées des noeuds de la grille DF 

    T = lib.solve_eqT(L,Hs,Hf,I,J,ks,kf,rhof,cf,umax,f,xl,xr,yd,yu,Te)
    T2d  = lib.convert_u1d2d(T,I,J)                      
    #print('T2D =',T2d)

    # Affichage 
    ## Iso-contour température Ts
    fig, ax = plt.subplots(figsize=(14.5, 80*Ht/L))
    Nblevels = 51
    im = plt.contourf(X, Y, T2d, Nblevels,cmap='hot' )
    fig.colorbar(im, location = 'right', shrink=0.9)

    ax.set_title('Temperature field. Masse volumique : ρf={rhof} kg/m^3'.format(rhof=rhof), fontsize=16)
    ax.set_xticks(np.linspace(0,L,5,endpoint=True))
    ax.set_yticks(np.linspace(-Hs,Hf,5,endpoint=True))
    plt.savefig(f'Iso_Temp_I{I}_J{J}_2.png', dpi=300)
    plt.show()

    ## Profil de température en x = 0. 0.25 0.5 0.75 1 
    x1 = 0.0
    x2 = 0.25
    x3 = 0.5
    x4 = 0.75
    x5 = 1.0
    T1,T2,T3,T4,T5 = lib.profil1dT(T2d,I,J,L,x1,x2,x3,x4,x5)
    plt.figure(figsize=(13, 4))
    plt.plot(T1,yn,label='x = x1') 
    plt.plot(T2,yn,label='x = x2') 
    plt.plot(T3,yn,label='x = x3')
    plt.plot(T4,yn,label='x = x4')
    plt.plot(T5,yn,label='x = x5')
    plt.title(f'Profil de température en fonction de y,(I={I}, J={J}). Masse volumique: ρf={rhof} kg/m^3')
    plt.xlabel('T [K]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.legend()
    plt.savefig(f'Profil_T_y_I{I}_J{J}_2.png', dpi=300)
    plt.show()
    
    plt.close('all')