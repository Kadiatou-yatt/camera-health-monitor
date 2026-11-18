def iou(box_a, box_b):
    # Format attendu : [x_min, y_min, x_max, y_max]
    
    # Déterminer les coordonnées de la boîte d'intersection
    x_inter_min = max(box_a[0], box_b[0])  
    y_inter_min = max(box_a[1], box_b[1]) 
    x_inter_max = min(box_a[2], box_b[2])  
    y_inter_max = min(box_a[3], box_b[3])  
    
    # 2. Calculer largeur et hauteur de l'intersection
    inter_largeur = max(0, x_inter_max - x_inter_min)
    inter_hauteur = max(0, y_inter_max - y_inter_min)
    aire_intersection = inter_largeur * inter_hauteur
    

    aire_box_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    aire_box_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    

    aire_union = aire_box_a + aire_box_b - aire_intersection
    
    # Sécurité anti-division par zéro 
    if aire_union == 0:
        return 0.0
        
    return aire_intersection / aire_union
