from fastapi import FastAPI, Depends, Query, HTTPException, Path, Body
from pydantic import BaseModel, Field
from sqlalchemy.orm import sessionmaker, Session
from app.database import SessionLocal, engine
from app.models.users import Base
from app.models.models import Villa
from typing import Annotated, Optional
from app.auth.jwt_bearer import jwtBearer
import app.auth.auth as auth
from app.auth.jwt_bearer import jwtBearer
import numpy as np
import pandas as pd  # Utilisé pour la manipulation de données
import joblib  # Utilisé pour charger le modèle sauvegardé



app = FastAPI()
app.include_router(auth.router)

#Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)

# Charger le modèle depuis le disque
modele = joblib.load('app/immo/model_prix_villa.pkl')


class VillaTopredict(BaseModel):
    superficie_m2: float = Field(..., gt=0, description="Superficie en mètres carrés")
    nombre_pieces: int = Field(..., ge=1, description="Nombre total de pièces")
    nombre_salles_bain: int = Field(..., ge=0, description="Nombre de salles de bain")

    jardin: int = Field(..., description="Présence d'un jardin")
    piscine: int = Field(..., description="Présence d'une piscine")
    parking: int = Field(..., description="Présence d'un parking")
    terrasse: int = Field(..., description="Présence d'une terrasse")
    cuisineEquipee: int = Field(..., description="Cuisine équipée")
    securisee: int = Field(..., description="Résidence sécurisée")

    balcon: int = Field(..., description="Présence d'un balcon")
    cite: int = Field(..., description="Situé dans une cité")
    meuble: int = Field(..., description="Villa meublée")

    basse: int = Field(..., description="Villa basse")
    duplex: int = Field(..., description="Villa duplex")
    triplex: int = Field(..., description="Villa triplex")
    
    
class VillaInDB(VillaTopredict):
    prix_fcfa: Optional[float] = Field(None, description="Prix de la villa en FCFA")
    prix_predit : int
    
def get_db ():
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

db = Annotated[Session, Depends(get_db)]  

def get_len_pred(db:db):
    return db.query(Villa).count()

lenpred = Annotated[int, Depends(get_len_pred)]  


@app.get("/")
def acceuil():
    return "Bienvenu sur l'estimateur immobilier"


# Estimer un bien immobilier
@app.post("/predict", dependencies=[Depends(jwtBearer())], tags=["Prediction"])
async def predict(villa : VillaTopredict, db:db):
    # Extraction et validation des données d'entrée en utilisant Pydantic
    villa_dict = villa.model_dump()
    donnees_df = pd.DataFrame([villa_dict])  # Conversion en DataFrame

    # Utilisation du modèle pour prédire et obtenir les probabilités
    predictions = modele.predict(donnees_df)  # Probabilité de la classe positive (diabète)

    # Compilation des résultats dans un dictionnaire
    resultats = villa_dict.copy()
    resultats['prix_predit'] = int(predictions[0])
    villa_db = VillaInDB(**resultats)
    # Enregistrement de la prediction
    new_prediction = Villa(
        #  Prix prédit par le modèle
            prix_predit = villa_db.prix_predit,
        
            #  Caractéristiques principales
            superficie_m2 = villa_db.superficie_m2,
            nombre_pieces = villa_db.nombre_pieces,
            nombre_salles_bain = villa_db.nombre_salles_bain,
        
            #  Équipements
            jardin = villa_db.jardin,
            piscine = villa_db.piscine,
            parking = villa_db.parking,
            terrasse = villa_db.terrasse,
            cuisineEquipee = villa_db.cuisineEquipee,
            securisee = villa_db.securisee,
        
            # Autres caractéristiques
            balcon = villa_db.balcon,
            cite = villa_db.cite,
            meuble = villa_db.meuble,
        
            # 🏗️ Type de bien
            basse = villa_db.basse,
            duplex = villa_db.duplex,
            triplex = villa_db.triplex
    )
    db.add(new_prediction)
    db.commit()


    # Renvoie les résultats sous forme de JSON
    return {"prix_predit": int(predictions[0])}


# Obetnir le resutat des estimations immobilière déjà réalisé
@app.get("/predictions/", tags=["Prediction"])
async def get_all_predictions(db:db,
                              lenpred : lenpred,
                              limit : Annotated[int, Query(description="Number of prediction you want")]=None, 
                              start : Annotated[int, Query()]=0 ):
    if not limit:
        limit = lenpred
    
    predictions = db.query(Villa).offset(start).limit(limit).all()
    if not predictions:
        return HTTPException(status_code=404, detail="Aucun post trouvé")
    return predictions


# Obetnir le resutat d'une estimation immobilière précise
@app.get("/predictions/{pred_id}/", tags=["Prediction"])
async def get_prediction_by_id(pred_id : Annotated[int, Path(ge=0)], db:db):
    prediction = db.query(Villa).filter(Villa.id == pred_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Aucune prédiction pour ce id")
    return prediction


# Marquer le vraie prix de la villa
@app.put("/predictions/{pred_id}/", tags=["Prediction"])
async def set_villa_price(pred_id : Annotated[int, Path(ge=0)], db:db, prix_fcfa:Annotated[int, Body(ge=0)]):
    prediction = db.query(Villa).filter(Villa.id == pred_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Aucune prédiction pour ce id")
    
    prediction.prix_fcfa = prix_fcfa
    db.commit()
    
    return {"message":"Le vrai prix de la villa {} a été bien marqué".format(pred_id)}
    


# Supprimer une estimation
@app.delete("/predictions/{pred_id}/", tags=["Prediction"])
async def set_villa_price(pred_id : Annotated[int, Path(ge=0)], db:db):
    prediction = db.query(Villa).filter(Villa.id == pred_id).first()
    if not prediction:
        raise HTTPException(status_code=404, detail="Aucune prédiction pour ce id")
    db.delete(prediction)
    return {"message":"La prediction {} a été bien supprimée".format(pred_id)}