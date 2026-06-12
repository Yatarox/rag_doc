import sys
import os



from chain import Chain

def test_chain():
    print("="*50)
    print("🧪 TEST DE LA CHAÎNE RAG")
    print("="*50)
    
    try:
        # 1. Initialisation
        print("\n1️⃣ Initialisation de la chaîne...")
        chain = Chain()
        print("✅ Chaîne initialisée avec succès")
        
        # 2. Vérification de ChromaDB
        print("\n2️⃣ Vérification de ChromaDB...")
        count = chain.chroma_db.collection.count()
        print(f"✅ {count} documents dans ChromaDB")
        
        if count == 0:
            print("\n⚠️  ChromaDB est vide !")
            print("   Ingère d'abord un PDF avec :")
            print("   python -c 'from ingest import Ingestor; Ingestor().ingest_init()'")
            return
        
        # 3. Test avec une question simple
        print("\n3️⃣ Test avec une question simple...")
        question = "De quoi parle ce document ?"
        print(f"📝 Question : {question}")
        
        print("🤖 Génération de la réponse...")
        reponse = chain.run(question)
        
        print("\n" + "="*50)
        print("📢 RÉPONSE :")
        print("="*50)
        print(reponse)
        print("="*50)
        
        # 4. Test avec une question plus précise (optionnel)
        print("\n4️⃣ Deuxième question...")
        question2 = "Quels sont les points principaux ?"
        print(f"📝 Question : {question2}")
        
        reponse2 = chain.run(question2)
        print(f"\n📢 Réponse : {reponse2}\n")
        
        print("✅ Tests terminés avec succès !")
        
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        import traceback
        traceback.print_exc()

def quick_test():
    """Test encore plus simple"""
    print("🚀 Test rapide...")
    
    try:
        from chain import Chain
        chain = Chain()
        
        # Vérifie juste si ChromaDB a des données
        count = chain.chroma_db.collection.count()
        print(f"✅ ChromaDB OK : {count} chunks")
        
        if count > 0:
            # Test une seule question
            reponse = chain.run("Résume le document en une phrase")
            print(f"\n📢 Résumé : {reponse[:200]}...")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    # Choix du test
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_test()
    else:
        test_chain()