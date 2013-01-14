Auteurs :  Quentin MAOUHOUB
            Florian BOUDINET

Date : Decembre 2012

Lieu : ENSEIRB-MATMECA

Architecture :

/DetectLine
        |
        |-/Videos    Dossier contenant les videos du projet
        |
        |

__________________________
_________Etat


Notre code contient tout un tas de fonctions. Nous avons essayé moult algo, solutions et autres idées afin de répondre aux problematiques.

detectLinePoints() Permet de detecter les lignes du terrain.

detectLinePoints_glitch() utilise le meme algorithme mais ne trouve pas les points d'intersection ligne/terrain (blanc/vert). Il se contente de peindre le vert, le blanc, le noir et (le bleu). Mais le resultat est assez concluant.



_______________________________
______Utilisation

$> make
$> ./DetectLine ${VIDEO_PATH}


Ou alors ouvrir le *.pro avec QtCreator.
