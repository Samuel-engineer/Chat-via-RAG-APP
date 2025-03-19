with open("links_file.txt", 'r') as file:  
    urls = [line.strip() for line in file] 

import os
from langchain_community.document_loaders import PlaywrightURLLoader

# Scraper chaque URL séparément pour ne pas perdre les données en cas d’échec
current_dir = os.path.dirname(__file__)  # Récupère le dossier du script
output_folder = os.path.join(current_dir, "..", "api")  # Aller vers "output"
output_file = os.path.join(output_folder, "result.txt")

# Nombre max de tentatives par page
MAX_RETRIES = 3  

# Stocker les résultats
successful_documents = []
failed_urls = []

def load_page_with_retry(url, retries=MAX_RETRIES):
    """Charge une page et relance Playwright en cas de crash."""
    attempt = 0
    while attempt < retries:
        try:
            print(f"🔄 Chargement de {url} (Tentative {attempt+1}/{retries})...")

            # Créer un nouveau PlaywrightURLLoader à chaque tentative
            loader = PlaywrightURLLoader(urls=[url], remove_selectors=["header","footer","div.lc-inline_column_first-edit","div.lc-inline_container-section-edit","a.FootnoteRef"])
            data = loader.load()
            
            print(f"✅ Succès : {url}")
            return data  # Retourne le document récupéré

        except Exception as e:
            print(f"❌ Erreur sur {url}: {e}")
            attempt += 1            
            if attempt >= retries:
                print(f"🚨 Échec final : {url}")
                return None  # Échec définitif après `MAX_RETRIES`

if os.path.exists(output_file):
    os.remove(output_file)

for url in urls:
    result = load_page_with_retry(url)
    if result:
        successful_documents.extend(result)  # Ajouter les données récupérées
    else:
        failed_urls.append(url)  # Stocker les URLs échouées
   
    with open(output_file, 'a') as file:
        file.write(result[0].page_content)

print(f"\n✅ {len(successful_documents)} pages chargées avec succès.")
print(f"❌ {len(failed_urls)} pages ont échoué : {failed_urls}")


# Écrire dans le fichier situé dans "output/"

print(f"Les URLs ont été enregistrées dans {output_file}")