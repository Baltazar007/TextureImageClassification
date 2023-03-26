from typing import List
# from imutils import paths
from os import listdir
kth_path = r"D:/TECCART/2023/1-SessionHiver/IA2/projetImage/datasets/TIPS2-a/"
# fil etape hedhir on prend les noms des dossier et on les mettre dans une liste de
kth_dir: List[str] = listdir(kth_path)  # lister le contenu d'un dossier
print(kth_dir)
