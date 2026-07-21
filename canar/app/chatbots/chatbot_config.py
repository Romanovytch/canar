import mimetypes

from pydantic import ValidationError
from sqlalchemy import JSON, Column
from sqlmodel import Field, Session, SQLModel, select


class ChatbotConfig(SQLModel, table=True):
    # id avec des minuscules, chiffres et underscores uniquement
    id: str = Field(primary_key=True, regex=r"^[a-z0-9_]+$")
    # name et system_prompt ne peuvent pas être vides
    name: str = Field(min_length=1)
    description: str
    system_prompt: str = Field(min_length=1)
    collections: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    accepted_file_types: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    export_extension: str | None = Field(default=None)
    model: str = Field(default="gpt-4o")
    # top_k strictement positif
    top_k: int = Field(default=5, ge=1)
    # score_threshold entre 0.0 et 1.0
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    # max_context_tokens cohérent avec le slider
    max_context_tokens: int = Field(default=2048, ge=256)

    allowed_tools: list[str] = Field(default=[], sa_column=Column(JSON))

    @property
    def exporte_mime_type(self) -> str:
        """"""
        if not self.export_extension:
            return "text/plain"

        guessed_type, _ = mimetypes.guess_type(f"dummy.{self.export_extension}")

        return guessed_type or "text/plain"

    def saveChatbot(self, session: Session, commit: bool = False) -> bool:
        """
        Sauvegarde l'état du chatbot dans le bdd en faisant un create or update
        Retourne True si l'opération réussit et que l'objet est valide
        """

        # Validation à introduire ici

        statement = select(ChatbotConfig).where(ChatbotConfig.id == self.id)
        existing_bot = session.exec(statement).first()

        if existing_bot:
            existing_bot.name = self.name
            existing_bot.description = self.description
            existing_bot.system_prompt = self.system_prompt
            existing_bot.collections = self.collections
            existing_bot.model = self.model
            existing_bot.top_k = self.top_k
            existing_bot.score_threshold = self.score_threshold
            existing_bot.max_context_tokens = self.max_context_tokens

            session.add(existing_bot)

        else:
            session.add(self)

        if commit:
            session.commit()
            if existing_bot:
                session.refresh(existing_bot)
            else:
                session.refresh(self)

        return True

    @classmethod
    def createChatbot(cls, yaml_data: dict) -> "ChatbotConfig":
        """
        Crée et retourne une instance d'objet ChatbotConfig à partir des données YAML.
        La sauvegarde en base de données est géré séparément.
        """
        try:
            return cls.model_validate(yaml_data)
        except Exception as e:
            raise ValueError(f"Erreur de synthaxe YAML pour le chatbot {e}") from e

    @classmethod
    def checkChatbot(cls, yaml_data: dict) -> bool:
        """
        Utilise Pydantic pour valider la structure YAML avant toute création.
        Retourne True si les données respectent les règles, False sinon.
        """
        try:
            cls.model_validate(yaml_data)
            return True
        except ValidationError as e:
            print(
                f"[Validation Échouée] Erreur pour le chatbot "
                f"'{yaml_data.get('id', 'inconnu')}' :\n{e}"
            )
            return False

    def deleteChatbot(self, session: Session, commit: bool = True):
        """"""
        existing_bot = session.get(ChatbotConfig, self.id)
        if existing_bot:
            session.delete(existing_bot)
            if commit:
                session.commit()
            return True

        return False
