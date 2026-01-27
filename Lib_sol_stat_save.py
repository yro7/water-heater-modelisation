def solve_eqT(L,Hs,Hf,I,J,ks,kf,rhof,cf,umax,f,xl,xr,yd,yu,Te) :

    import numpy as np
    import numpy.linalg as npl
    import scipy as sparse
    import scipy.sparse as spsp
    import scipy.sparse.linalg
    
    dx = L/(I-1)                        # Pas de maillage en x 
    dy = (Hs+Hf)/(J-1)                  # Pas de maillage en y
    xn = np.linspace(0., L, I)          # Abscisses des noeuds 
    yn = np.linspace(-Hs, Hf, J)        # Ordonnées des noeuds  
    X,Y = np.meshgrid(xn, yn)           # Coordonnées des noeuds de la grille DF 
   
    
    # Calcul de la vitesse aux noeuds du maillage
    uf = vitesse_fluide(J,dy,yn,Hf,umax)
    #print('uf = ', uf) 
    
    # Calcul du terme source volumique d'énergie aux noeuds du maillage
    Se = source_energie(I,J,xn,yn,xl,xr,yd,yu,f)
    #print('Se = ', Se) 
    
    # Assemblage de la matrice associée au problème 
    A = Matrice(I,J,dx,dy,yn,ks,kf,uf,rhof,cf)
    #print('A = ',A)
    
    # Assemblage du second membre associé au problème 
    S = Second_membre(I,J,dx,dy,yn,Se,Te)
    
    # Calcul de la solution du problème 
    T = spsp.linalg.spsolve(A, S, permc_spec=None, use_umfpack=True)
    
    return T 

def Matrice(I,J,dx,dy,yn,ks,kf,uf,rhof,cf):   
    import numpy as np
    import numpy.linalg as npl
    import scipy as sparse
    import scipy.sparse as spsp
    import scipy.sparse.linalg
    
    K = I*J
    # On déclare une matrice creuse de dimension K x K
    A = spsp.csr_matrix((K, K))
    # Assemblage de la matrice --> Calcul des A(l,l') non nuls
    # Cas des noeuds internes 
    for i in range(1,I-1):
        for j in  range(1,J-1):
            l = indk(i,j,I)
            le = indk(i+1,j,I)
            lo = indk(i-1,j,I)
            ln = indk(i,j+1,I)
            ls = indk(i,j-1,I)
           
            # A compléter  
            # Initialisation par défaut des conductivités 
            if yn[j]< 0 :
                kl = ks 
                kln = ks
                kls = ks
            else : 
                kl = kf
                kln = kf
                kls = kf

            # Cas particuliers (Interface Solide/Fluide)
            if yn[j]<= 0.5*dy and yn[j] >= -0.5*dy : 
                kl = 0.5*kf*(1.0-yn[j]/dy)+0.5*ks*(1.0+yn[j]/dy)

            if yn[j]<=0 and yn[j+1] >= 0 :   # cas yn \in [-dy,0]
                dyl = - yn[j] 
                dyn = yn[j+1]
                kln = dy*kf*ks/(dyl*kf+dyn*ks)

            if yn[j]>=0 and yn[j-1] <= 0 :   # cas yn \in [0,dy]
                dyl = yn[j] 
                dys = -yn[j-1]
                kls = dy*kf*ks/(dyl*ks+dys*kf)
            
            # Terme de convection (Schéma Upwind: flux vient de l'ouest car u>=0)
            conv = rhof*cf*uf[j]/dx

            # Remplissage de la matrice pour les nœuds internes
            # Bilan : Convection + Diffusion_x + Diffusion_y = Source
            # Diagonale
            A[l,l] = 2.*kl/dx**2 + (kln+kls)/dy**2 + conv
            # Voisins
            A[l,le] = -kl/dx**2
            A[l,lo] = -kl/dx**2 - conv
            A[l,ln] = -kln/dy**2
            A[l,ls] = -kls/dy**2
            
    # (Fin à compléter)            
    # Cas des condtions aux limites
    # Boucle sur les noeuds de la frontière x=0 (entrée du fluide)
    for j in  range(1,J-1):
        i=0
        l = indk(i,j,I)
        le = indk(i+1,j,I)
        ln = indk(i,j+1,I)
        ls = indk(i,j-1,I)
        # Initialisation par défaut des conductivités 
        if yn[j]< 0 :
            kl = ks 
            kln = ks
            kls = ks
        else : 
            kl = kf
            kln = kf
            kls = kf
        # Cas particuliers 
        if yn[j]<= 0.5*dy and yn[j] >= -0.5*dy : 
            kl = 0.5*kf*(1.0-yn[j]/dy)+0.5*ks*(1.0+yn[j]/dy)
        if yn[j]<=0 and yn[j+1] >= 0 :   # cas yn \in [-dy,0]
            dyl = - yn[j] 
            dyn = yn[j+1]
            kln = dy*kf*ks/(dyl*kf+dyn*ks)
        if yn[j]>=0 and yn[j-1] <= 0 :   # cas yn \in [0,dy]
            dyl = yn[j] 
            dys = -yn[j-1]
            kls = dy*kf*ks/(dyl*ks+dys*kf)
        # Prise en compte de la condition aux limite en x = 0 
        if yn[j] >=0  :    # on est dans le fluide (T = Te) 
            A[l,l] = 1.0
        else :                  # on est dans le solide (flux nul)
           A[l,l] = kl/dx**2 + (kln+kls)/dy**2 
           A[l,le] = -kl/dx**2
           A[l,ls] = -kls/dy**2  
           A[l,ln] = -kln/dy**2 
    # Boucle sur les noeuds de la frontière x=L  (flux conductif nul)
    for j in  range(1,J-1):
        i = I-1
        l = indk(i,j,I)
        lo = indk(i-1,j,I)
        ln = indk(i,j+1,I)
        ls = indk(i,j-1,I)
        # Initialisation par défaut des conductivités 
        if yn[j]< 0 :
            kl = ks 
            kln = ks
            kls = ks
        else : 
            kl = kf
            kln = kf
            kls = kf
        # Cas particuliers 
        if yn[j]<= 0.5*dy and yn[j] >= -0.5*dy : 
            kl = 0.5*kf*(1.0-yn[j]/dy)+0.5*ks*(1.0+yn[j]/dy)
        if yn[j]<=0 and yn[j+1] >= 0 :  # cas yn \in [-dy,0]
            dyl = - yn[j] 
            dyn = yn[j+1]
            kln = dy*kf*ks/(dyl*kf+dyn*ks)
        if yn[j]>=0 and yn[j-1] <= 0 :  # cas yn \in [0,dy]
            dyl = yn[j] 
            dys = -yn[j-1]
            kls = dy*kf*ks/(dyl*ks+dys*kf)
        A[l,l] = kl/dx**2 + (kln+kls)/dy**2 + rhof*cf*uf[j]/dx
        A[l,lo] = -kl/dx**2 - rhof*cf*uf[j]/dx
        A[l,ls] = -kls/dy**2  
        A[l,ln] = -kln/dy**2 
    # Boucle sur les noeuds de la frontière y=Hf (symétrie, flux nul)
    for i in  range(1,I-1):
        j=J-1
        l = indk(i,j,I)
        le = indk(i+1,j,I)
        lo = indk(i-1,j,I)
        ls = indk(i,j-1,I)
        kl = kf
        kls = kf
        A[l,l] = 2.*kl/dx**2 + kls/dy**2 + rhof*cf*uf[j]/dx
        A[l,le] = -kl/dx**2
        A[l,lo] = -kl/dx**2 - rhof*cf*uf[j]/dx
        A[l,ls] = -kls/dy**2     
    # Boucle sur les noeuds de la frontière y=-Hs (flux nul)
    for i in range(1,I-1):
        j = 0
        l = indk(i,j,I)
        le = indk(i+1,j,I)
        lo = indk(i-1,j,I)
        ln = indk(i,j+1,I)
        kl = ks
        kln = ks
        A[l,l] = 2.*kl/dx**2 + kln/dy**2 
        A[l,le] = -kl/dx**2
        A[l,lo] = -kl/dx**2 
        A[l,ln] = -kln/dy**2 
    # Cas particulier des coins 
    # Coin en x=0,y=Hf
    i = 0
    j = J-1 
    l = indk(i,j,I)
    A[l,l] = 1.0
    # Coin en x=L,y=Hf
    i = I-1
    j = J-1 
    l = indk(i,j,I)
    lo = indk(i-1,j,I)
    ls = indk(i,j-1,I)
    kl = kf
    kls = kf
    A[l,l] = kl/dx**2 + kls/dy**2  + rhof*cf*uf[j]/dx
    A[l,lo] = -kl/dx**2 - rhof*cf*uf[j]/dx
    A[l,ls] = -kls/dy**2 
    # Coin en x=0,y=-Hs
    i = 0
    j = 0
    l = indk(i,j,I)
    le = indk(i+1,j,I)
    ln = indk(i,j+1,I)
    kln = ks
    kl = ks
    A[l,l]  = kl/dx**2 + kln/dy**2  
    A[l,le] = -kl/dx**2
    A[l,ln] = -kln/dy**2 
    # Coin en x=L,y=-Hs
    i = I-1
    j = 0
    l = indk(i,j,I)
    lo = indk(i-1,j,I)
    ln = indk(i,j+1,I)
    kln = ks
    kl = ks
    A[l,l] = ks/dx**2 + kln/dy**2  
    A[l,lo] = -kl/dx**2
    A[l,ln] = -kln/dy**2 
    return A



def Second_membre(I,J,dx,dy,yn,Se,Te):
    import numpy as np    
    K = I*J
    # Initialisation du second membre S 
    S =  np.zeros((K,1))
    # Boucle sur tous les noeuds pour la contribution de la source f à S 
    
    # A compléter 
    for i in range(I):
        for j in range(J):
            l = i + j*I
            S[l, 0] = Se[i, j]   
    # Boucle sur les noeuds de la frontière x=0 (Température imposée : T = Te)
    # (fin à completér)
    # A compléter 
    for j in range(J):
        if yn[j] >= 0: # Condition de Dirichlet uniquement dans le fluide
            l = 0 + j*I # i=0
            S[l, 0] = Te
    # (fin à compléter)
    return S 

def indk(i,j,I) :
    l = i + j*I
    return l
    
def vitesse_fluide(J,dy,yn,Hf,umax):
    import numpy as np
    uf = np.zeros((J,1))
    for j in range(J):
        if yn[j]>-0.5*dy:
            uf[j] = umax*(2.0*yn[j]/Hf - (yn[j]/Hf)**2)
    return uf

    
def source_energie(I,J,xn,yn,xl,xr,yd,yu,f):
    import numpy as np
    Se = np.zeros((I,J))
    for i in range(I):
        for j in range(J):
            if yn[j] < yu and yn[j] > yd : 
                if xn[i] < xr and xn[i] > xl :
                    Se[i,j] = f
    return Se
    
def convert_u1d2d(u,I,J):
    import numpy as np
    u2d = np.zeros((I,J))
    for i in range(I):
        for j in  range(J):
            l = indk(i,j,I)
            u2d[i,j] = u[l]
    return np.transpose(u2d)

def profil1dT(T2d,I,J,L,x1,x2,x3,x4,x5):
    import numpy as np
    T2d = np.transpose(T2d)
    from math import floor
    i1 = floor(x1*I/L)
    i2 = floor(x2*I/L)
    i3 = floor(x3*I/L)
    i4 = floor(x4*I/L)
    i5 = floor(x5*I/L)-1
    T1 = T2d[i1,:]
    T2 = T2d[i2,:]
    T3 = T2d[i3,:]
    T4 = T2d[i4,:]
    T5 = T2d[i5,:]
    return T1,T2,T3,T4,T5
