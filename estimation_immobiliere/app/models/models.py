from sqlalchemy import Column, Integer, Float, String
from ..database import Base


class Villa(Base):
    __tablename__ = "villas"

    id = Column(Integer, primary_key=True, index=True)

    #  Prix réel (optionnel si prédiction)
    prix_fcfa = Column(Float, nullable=True)

    #  Prix prédit par le modèle
    prix_predit = Column(Float, nullable=True)

    #  Caractéristiques principales
    superficie_m2 = Column(Float, nullable=False)
    nombre_pieces = Column(Integer, nullable=False)
    nombre_salles_bain = Column(Integer, nullable=False)

    #  Équipements
    jardin = Column(Integer, default=False)
    piscine = Column(Integer, default=False)
    parking = Column(Integer, default=False)
    terrasse = Column(Integer, default=False)
    cuisineEquipee = Column(Integer, default=False)
    securisee = Column(Integer, default=False)

    # Autres caractéristiques
    balcon = Column(Integer, default=False)
    cite = Column(Integer, default=False)
    meuble = Column(Integer, default=False)

    # 🏗️ Type de bien
    basse = Column(Integer, default=False)
    duplex = Column(Integer, default=False)
    triplex = Column(Integer, default=False)