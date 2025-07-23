import asyncio
import aiohttp
from typing import List, Dict, Optional
from datetime import datetime
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MLQuestion:
    id: str
    text: str
    date_created: datetime
    status: str
    item_id: str
    from_user_id: str
    answer: Optional[str] = None
    answer_date: Optional[datetime] = None


@dataclass
class MLItem:
    id: str
    title: str
    price: float
    currency_id: str
    condition: str
    category_id: str
    permalink: str
    status: str
    description: Optional[str] = None


class MercadoLibreIntegration:
    def __init__(self, access_token: str, user_id: str):
        self.access_token = access_token
        self.user_id = user_id
        self.base_url = "https://api.mercadolibre.com"
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    async def _request_public(
        self, url: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 401:
                    logger.error(
                        "Unauthorized (401) - Verifica si el token es válido o si el usuario tiene publicaciones visibles."
                    )
                else:
                    logger.error(f"Request failed ({response.status}) - {url}")
                return None
        except Exception as e:
            logger.error(f"Exception in public request: {e}")
            return None

    async def _get_items_via_public_search(self, limit: int) -> List[MLItem]:
        try:
            user_info_url = f"{self.base_url}/users/{self.user_id}"
            site_id = "MLA"
            user_data = await self._request_public(user_info_url)
            if user_data and user_data.get("site_id"):
                site_id = user_data["site_id"]

            url = f"{self.base_url}/sites/{site_id}/search"
            params = {"seller_id": self.user_id, "limit": min(limit, 50), "offset": 0}

            logger.info(f"Using public search fallback: {url}")
            data = await self._request_public(url, params)
            if not data:
                return []

            items = []
            for item_data in data.get("results", []):
                try:
                    item = MLItem(
                        id=item_data["id"],
                        title=item_data["title"],
                        price=float(item_data.get("price", 0)),
                        currency_id=item_data.get("currency_id", ""),
                        condition=item_data.get("condition", ""),
                        category_id=item_data.get("category_id", ""),
                        permalink=item_data.get("permalink", ""),
                        status=item_data.get("status", "active"),
                    )
                    items.append(item)
                except Exception as e:
                    logger.warning(f"Error processing item: {e}")
                    continue

            return items

        except Exception as e:
            logger.error(f"Error in public search fallback: {e}")
            return []

    async def validate_token(self) -> bool:
        """Valida si el token de acceso es válido y tiene permisos."""
        url = f"{self.base_url}/users/me"
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(url, headers=self._get_headers()) as response:
                if response.status == 200:
                    logger.info("Token validation successful")
                    return True
                else:
                    logger.warning(
                        f"Token validation failed ({response.status}) - continuing with limited functionality"
                    )
                    return False
        except Exception as e:
            logger.error(f"Error validating token: {e}")
            return False

    async def get_user_items(self, limit: int = 50) -> List[MLItem]:
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Try to validate token but don't block if it fails
            token_valid = await self.validate_token()
            if token_valid:
                logger.info("Token validation successful - using full API access")
            else:
                logger.warning(
                    "Token validation failed - attempting limited functionality"
                )

            logger.info("Trying private API to get user items")
            items = await self._get_items_via_private_api(limit)

            if items:
                logger.info(f"Successfully got {len(items)} items via private API")
                return items

            logger.info("Private API failed, using public search API fallback")
            return await self._get_items_via_public_search(limit)

        except Exception as e:
            logger.error(f"Error getting user items: {e}")
            return []

    async def _get_items_via_private_api(self, limit: int) -> List[MLItem]:
        try:
            url = f"{self.base_url}/users/{self.user_id}/items/search"
            params = {"limit": min(limit, 50), "offset": 0, "status": "active"}

            async with self.session.get(
                url, headers=self._get_headers(), params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    items = []
                    results = data.get("results", [])
                    logger.info(f"Found {len(results)} items via private API")

                    # Try to get details for each item, but create basic items even if details fail
                    for item_id in results:
                        item_detail = await self._get_item_details(item_id)
                        if item_detail:
                            items.append(item_detail)
                        else:
                            # Create basic item with just ID if details fail
                            logger.info(
                                f"Creating basic item for {item_id} (details unavailable)"
                            )
                            basic_item = MLItem(
                                id=item_id,
                                title=f"Item {item_id}",
                                price=0.0,
                                currency_id="ARS",
                                condition="unknown",
                                category_id="unknown",
                                permalink=f"https://articulo.mercadolibre.com.ar/{item_id}",
                                status="active",
                            )
                            items.append(basic_item)

                    return items
                else:
                    logger.warning(f"Private API failed with status: {response.status}")
                    if response.status == 401:
                        logger.warning("Token unauthorized for private API")
                    elif response.status == 403:
                        logger.warning("Token lacks permissions for private API")
                    return []

        except Exception as e:
            logger.error(f"Error in private API: {e}")
            return []

    async def _get_item_details(self, item_id: str) -> Optional[MLItem]:
        try:
            url = f"{self.base_url}/items/{item_id}"
            # Add authentication headers - this was missing!
            async with self.session.get(url, headers=self._get_headers()) as response:
                if response.status == 200:
                    item_data = await response.json()
                    logger.info(f"Successfully got details for item {item_id}: {item_data['title']}")
                    return MLItem(
                        id=item_data["id"],
                        title=item_data["title"],
                        price=float(item_data.get("price", 0)),
                        currency_id=item_data.get("currency_id", ""),
                        condition=item_data.get("condition", "unknown"),
                        category_id=item_data.get("category_id", ""),
                        permalink=item_data.get("permalink", ""),
                        status=item_data.get("status", "active"),
                    )
                else:
                    logger.warning(
                        f"Could not get details for item {item_id}: {response.status}"
                    )
                    return None
        except Exception as e:
            logger.error(f"Error getting item details for {item_id}: {e}")
            return None

    # Método para obtener las preguntas de los artículos
    async def get_item_questions(
        self, item_id: str, status: str = "UNANSWERED"
    ) -> List[MLQuestion]:
        """Get questions for a specific item"""
        try:
            url = f"{self.base_url}/questions/search"
            params = {
                "item": item_id,
                "status": status,
                "sort": "date_created",
                "order": "desc",
            }

            async with self.session.get(
                url, headers=self._get_headers(), params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data:
                        logger.warning(
                            f"Empty response for questions on item {item_id}"
                        )
                        return []

                    questions = []

                    for q_data in data.get("questions", []):
                        try:
                            answer_data = q_data.get("answer")
                            answer_text = answer_data["text"] if answer_data and "text" in answer_data else None
                            answer_date = (
                                datetime.fromisoformat(answer_data["date_created"].replace("Z", "+00:00"))
                                if answer_data and "date_created" in answer_data else None
                            )
                            question = MLQuestion(
                                id=str(q_data["id"]),
                                text=q_data["text"],
                                date_created=datetime.fromisoformat(
                                    q_data["date_created"].replace("Z", "+00:00")
                                ),
                                status=q_data["status"],
                                item_id=item_id,
                                from_user_id=str(q_data["from"]["id"]),
                                answer=answer_text,
                                answer_date=answer_date,
                            )
                            questions.append(question)
                        except Exception as e:
                            logger.warning(f"Error processing question data: {e}")
                            continue

                    logger.info(f"Found {len(questions)} questions for item {item_id}")
                    return questions
                elif response.status == 403:
                    logger.warning(
                        f"Access denied for questions API on item {item_id} - token lacks permissions"
                    )
                    return []
                elif response.status == 404:
                    logger.info(f"No questions found for item {item_id} (404)")
                    return []
                else:
                    logger.error(
                        f"Error fetching questions for item {item_id}: {response.status}"
                    )
                    error_text = await response.text()
                    logger.error(f"Error details: {error_text}")
                    return []

        except Exception as e:
            logger.error(f"Error getting questions for item {item_id}: {e}")
            return []

    async def get_item_questions(
        self, item_id: str, status: str = "UNANSWERED"
    ) -> List[MLQuestion]:
        """Get questions for a specific item"""
        try:
            url = f"{self.base_url}/questions/search"
            params = {
                "item": item_id,
                "status": status,
                "sort": "date_created",
                "order": "desc",
            }

            async with self.session.get(
                url, headers=self._get_headers(), params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if not data:
                        logger.warning(
                            f"Empty response for questions on item {item_id}"
                        )
                        return []

                    questions = []

                    for q_data in data.get("questions", []):
                        try:
                            answer_data = q_data.get("answer")
                            answer_text = answer_data["text"] if answer_data and "text" in answer_data else None
                            answer_date = (
                                datetime.fromisoformat(answer_data["date_created"].replace("Z", "+00:00"))
                                if answer_data and "date_created" in answer_data else None
                            )
                            question = MLQuestion(
                                id=str(q_data["id"]),
                                text=q_data["text"],
                                date_created=datetime.fromisoformat(
                                    q_data["date_created"].replace("Z", "+00:00")
                                ),
                                status=q_data["status"],
                                item_id=item_id,
                                from_user_id=str(q_data["from"]["id"]),
                                answer=answer_text,
                                answer_date=answer_date,
                            )
                            questions.append(question)
                        except Exception as e:
                            logger.warning(f"Error processing question data: {e}")
                            continue

                    logger.info(f"Found {len(questions)} questions for item {item_id}")
                    return questions
                elif response.status == 403:
                    logger.warning(
                        f"Access denied for questions API on item {item_id} - token lacks permissions"
                    )
                    return []
                elif response.status == 404:
                    logger.info(f"No questions found for item {item_id} (404)")
                    return []
                else:
                    logger.error(
                        f"Error fetching questions for item {item_id}: {response.status}"
                    )
                    error_text = await response.text()
                    logger.error(f"Error details: {error_text}")
                    return []

        except Exception as e:
            logger.error(f"Error getting questions for item {item_id}: {e}")
            return []

    async def get_all_unanswered_questions(self) -> List[MLQuestion]:
        """Get all unanswered questions across all user items"""
        try:
            items = await self.get_user_items()
            all_questions = []

            for item in items:
                questions = await self.get_item_questions(item.id, "UNANSWERED")
                all_questions.extend(questions)

            # Sort by date (oldest first for priority)
            all_questions.sort(key=lambda x: x.date_created)

            return all_questions

        except Exception as e:
            logger.error(f"Error getting all unanswered questions: {e}")
            return []

    async def answer_question(self, question_id: str, answer_text: str) -> bool:
        """Answer a specific question"""
        try:
            url = f"{self.base_url}/answers"
            data = {"question_id": int(question_id), "text": answer_text}

            async with self.session.post(
                url, headers=self._get_headers(), json=data
            ) as response:
                response_text = await response.text()
                logger.info(f"Answer API response status: {response.status}")
                logger.info(f"Answer API response body: {response_text}")
                if response.status in [200, 201]:
                    logger.info(f"Successfully answered question {question_id}")
                    return True
                else:
                    logger.error(
                        f"Error answering question {question_id}: {response.status} - {response_text}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Error answering question {question_id}: {e}")
            return False

    async def get_item_context_for_ai(self, item_id: str) -> str:
        """Get comprehensive item context for AI responses"""
        try:
            # Try to get detailed item info
            item_detail = await self._get_item_details(item_id)

            if item_detail and item_detail.title != f"Item {item_id}":
                # We have real item details
                context = f"""
PRODUCTO: {item_detail.title}
PRECIO: {item_detail.price} {item_detail.currency_id}
CONDICIÓN: {item_detail.condition}
ESTADO: {item_detail.status}
ENLACE: {item_detail.permalink}
"""
            else:
                # Basic item info only
                context = f"""
PRODUCTO: Artículo ID {item_id}
ESTADO: Activo
ENLACE: https://articulo.mercadolibre.com.ar/{item_id}
NOTA: Información limitada disponible
"""

            return context.strip()

        except Exception as e:
            logger.error(f"Error getting item context: {e}")
            return f"PRODUCTO: Artículo ID {item_id}"

    async def process_unanswered_questions_with_ai(
        self, openai_integration, profile
    ) -> List[Dict]:
        """Process all unanswered questions and generate AI responses"""
        try:
            questions = await self.get_all_unanswered_questions()
            processed_questions = []

            logger.info(f"Processing {len(questions)} unanswered questions with AI")

            for question in questions:
                try:
                    # Get item context
                    item_context = await self.get_item_context_for_ai(question.item_id)

                    # Prepare AI prompt
                    ai_prompt = f"""
Como vendedor en MercadoLibre, responde esta pregunta de un cliente de manera profesional y útil.

CONTEXTO DEL PRODUCTO:
{item_context}

PREGUNTA DEL CLIENTE: {question.text}

Instrucciones:
- Responde de manera amigable y profesional
- Usa la información del producto para dar una respuesta precisa
- Si no tienes información suficiente, ofrece contactar por mensaje privado
- Mantén la respuesta concisa pero completa
- No inventes información que no esté en el contexto del producto
"""

                    # Get AI response
                    ai_response = await openai_integration.ask(
                        ai_prompt,
                        item_context,
                        "",  # No previous messages for ML questions
                        profile,
                    )

                    processed_questions.append(
                        {
                            "question": question,
                            "suggested_answer": ai_response,
                            "item_context": item_context,
                        }
                    )

                    logger.info(f"Generated AI response for question {question.id}")

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error processing question {question.id}: {e}")
                    continue

            return processed_questions

        except Exception as e:
            logger.error(f"Error processing unanswered questions: {e}")
            return []

    async def auto_answer_questions_with_ai(
        self, openai_integration, profile, auto_send: bool = False
    ) -> Dict:
        """Automatically process and optionally answer questions with AI"""
        try:
            processed = await self.process_unanswered_questions_with_ai(
                openai_integration, profile
            )
            results = {
                "total_questions": len(processed),
                "answered": 0,
                "failed": 0,
                "responses": [],
            }

            for item in processed:
                question = item["question"]
                suggested_answer = item["suggested_answer"]

                result_item = {
                    "question_id": question.id,
                    "question_text": question.text,
                    "suggested_answer": suggested_answer,
                    "answered": False,
                    "error": None,
                }

                if auto_send:
                    # Actually send the answer
                    success = await self.answer_question(question.id, suggested_answer)
                    if success:
                        results["answered"] += 1
                        result_item["answered"] = True
                    else:
                        results["failed"] += 1
                        result_item["error"] = "Failed to send answer"

                results["responses"].append(result_item)

            return results

        except Exception as e:
            logger.error(f"Error in auto answer process: {e}")
            return {
                "total_questions": 0,
                "answered": 0,
                "failed": 0,
                "error": str(e),
                "responses": [],
            }
