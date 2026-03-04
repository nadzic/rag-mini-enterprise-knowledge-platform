import logging

import inngest
from dotenv import load_dotenv

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag-mini-enterprise-knowledge-platform",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)
