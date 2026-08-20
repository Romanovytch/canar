class ToolError(Exception):
    """
    Classe de base pour toutes les exceptions liées au domaine des outils CanaR.
    """

    pass


class ProviderUnavailable(ToolError):
    """
    Levée lorsqu'un fournisseur d'outils (ex: serveur MCP) est indisponible ou injoignable.
    """

    def __init__(self, provider_id: str, reason: str = ""):
        self.provider_id = provider_id
        self.reason = reason
        msg = f"Le provider d'outils '{provider_id}' est indisponible"
        if reason:
            msg += f" : {reason}"
        super().__init__(msg)


class ToolNotFound(ToolError):
    """
    Levée lorsqu'un outil demandé est introuvable chez le provider ou dans le registre.
    """

    def __init__(self, tool_name: str, provider_id: str | None = None):
        self.tool_name = tool_name
        self.provider_id = provider_id
        if provider_id:
            msg = f"L'outil '{tool_name}' est introuvable sur le provider '{provider_id}'."
        else:
            msg = f"L'outil '{tool_name}' est introuvable."
        super().__init__(msg)


class InvalidToolArguments(ToolError):
    """
    Levée lorsque les arguments fournis par le LLM ne respectent pas le schéma de l'outil.
    """

    def __init__(self, tool_name: str, reason: str):
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Arguments invalides pour l'outil '{tool_name}' : {reason}")


class ToolExecutionError(ToolError):
    """
    Levée lorsqu'une erreur inattendue survient pendant l'exécution d'un outil.
    """

    def __init__(self, tool_name: str, original_error: Exception | str):
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Erreur d'exécution de l'outil '{tool_name}' : {original_error}")
