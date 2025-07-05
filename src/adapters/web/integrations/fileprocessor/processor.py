import base64
import io
import requests
from typing import List, Dict, Optional, Any, Tuple
from PIL import Image, ImageEnhance, ImageFilter
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from src.core.platform.config.service import ConfigurationService
from openai import AsyncOpenAI
import asyncio
from dataclasses import dataclass
import cv2
import numpy as np
from skimage import feature, filters, measure
import matplotlib.pyplot as plt
from io import BytesIO

@dataclass
class ImageSearchResult:
    page_number: int
    image_base64: str
    description: str
    confidence: float
    bbox: Optional[Dict[str, float]] = None  

class FileImageProcessor:
    def __init__(self, config_service: ConfigurationService):
        config = config_service.openai_config
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.vision_model = config.model  # Usar el modelo regular de DeepSeek
        
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def _analyze_image_with_python_tools(self, image_base64: str) -> Dict[str, Any]:
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))

            img_array = np.array(image)

            analysis = {
                'dimensions': image.size,
                'mode': image.mode,
                'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info,
                'dominant_colors': self._get_dominant_colors(img_array),
                'brightness': self._calculate_brightness(img_array),
                'contrast': self._calculate_contrast(img_array),
                'sharpness': self._calculate_sharpness(img_array),
                'text_detected': self._detect_text_regions(img_array),
                'objects_detected': self._detect_objects(img_array),
                'quality_score': self._calculate_quality_score(img_array)
            }
            
            return analysis
            
        except Exception as e:
            return {'error': f'Error analizando imagen: {str(e)}'}
    
    def _get_dominant_colors(self, img_array: np.ndarray) -> List[List[int]]:
        try:
            pixels = img_array.reshape(-1, 3) if len(img_array.shape) == 3 else img_array.reshape(-1, 1)

            from sklearn.cluster import KMeans

            sample_size = min(1000, len(pixels))
            sample_pixels = pixels[::len(pixels)//sample_size]

            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(sample_pixels)

            colors = kmeans.cluster_centers_.astype(int)
            return colors.tolist()

        except ImportError:
            return self._get_dominant_colors_fallback(img_array)
        except Exception:
            return []

    def _get_dominant_colors_fallback(self, img_array: np.ndarray) -> List[List[int]]:
        """Método alternativo para obtener colores dominantes sin sklearn"""
        try:
            # Convertir a PIL Image si es necesario
            if isinstance(img_array, np.ndarray):
                if len(img_array.shape) == 3:
                    image = Image.fromarray(img_array)
                else:
                    image = Image.fromarray(img_array, mode='L')
            else:
                image = img_array
            
            # Reducir a paleta de colores
            image_reduced = image.quantize(colors=5)
            palette = image_reduced.getpalette()
            
            # Extraer colores dominantes
            colors = []
            for i in range(5):
                color = palette[i*3:(i+1)*3]
                colors.append(color)
            
            return colors
        except Exception:
            return []
    
    def _calculate_brightness(self, img_array: np.ndarray) -> float:
        """Calcula el brillo promedio de una imagen"""
        try:
            if len(img_array.shape) == 3:
                # Convertir a escala de grises
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            return float(np.mean(gray))
        except Exception:
            # Método alternativo sin OpenCV
            if len(img_array.shape) == 3:
                # Conversión manual a escala de grises
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            return float(np.mean(gray))
    
    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        """Calcula el contraste de una imagen"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            return float(np.std(gray))
        except Exception:
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            return float(np.std(gray))
    
    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """Calcula la nitidez de una imagen usando el operador Laplaciano"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Aplicar operador Laplaciano
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except Exception:
            # Método alternativo sin OpenCV
            if len(img_array.shape) == 3:
                gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = img_array
            
            # Operador Laplaciano manual
            kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
            laplacian = cv2.filter2D(gray, cv2.CV_64F, kernel)
            return float(laplacian.var())
    
    def _detect_text_regions(self, img_array: np.ndarray) -> bool:
        """Detecta si hay regiones de texto en la imagen"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Aplicar filtro para detectar bordes
            edges = cv2.Canny(gray, 50, 150)
            
            # Buscar contornos rectangulares (posibles texto)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Contar contornos que podrían ser texto
            text_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 100 < area < 10000:  # Filtrar por área
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 < aspect_ratio < 5:  # Proporción típica de texto
                        text_contours += 1
            
            return text_contours > 5
            
        except Exception:
            return False
    
    def _detect_objects(self, img_array: np.ndarray) -> int:
        """Detecta el número aproximado de objetos en la imagen"""
        try:
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Aplicar umbralización
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Buscar contornos
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área
            significant_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Filtrar objetos pequeños
                    significant_contours += 1
            
            return significant_contours
            
        except Exception:
            return 0
    
    def _calculate_quality_score(self, img_array: np.ndarray) -> float:
        """Calcula un score de calidad de imagen basado en varios factores"""
        try:
            brightness = self._calculate_brightness(img_array)
            contrast = self._calculate_contrast(img_array)
            sharpness = self._calculate_sharpness(img_array)
            
            # Normalizar valores
            brightness_score = 1.0 - abs(brightness - 127.5) / 127.5  # Óptimo alrededor de 127.5
            contrast_score = min(contrast / 50.0, 1.0)  # Contraste alto es mejor
            sharpness_score = min(sharpness / 1000.0, 1.0)  # Nitidez alta es mejor
            
            # Calcular score promedio
            quality_score = (brightness_score + contrast_score + sharpness_score) / 3.0
            
            return float(quality_score)
            
        except Exception:
            return 0.5  # Score neutral si hay error
    
    async def search_images_in_pdf(
        self, 
        pdf_url: str, 
        search_term: str,
        max_results: int = 5
    ) -> List[ImageSearchResult]:
        """
        Busca imágenes en un PDF basándose en un término de búsqueda
        
        Args:
            pdf_url: URL del PDF a procesar
            search_term: Término a buscar (ej: "acolchado")
            max_results: Número máximo de resultados
            
        Returns:
            Lista de resultados de búsqueda de imágenes
        """
        try:
            # Descargar el PDF
            response = requests.get(pdf_url)
            response.raise_for_status()
            pdf_data = response.content
            
            # Extraer imágenes del PDF
            images_data = await self._extract_images_from_pdf(pdf_data)
            
            # Buscar imágenes relevantes usando OpenAI Vision
            results = await self._search_relevant_images(images_data, search_term, max_results)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error procesando PDF: {str(e)}")
    
    async def _extract_images_from_pdf(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Extrae todas las imágenes de un PDF"""
        images_data = []
        
        # Método 1: Usar PyMuPDF para extraer imágenes embebidas
        pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extraer imágenes embebidas
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Obtener el objeto de imagen
                    xref = img[0]
                    pix = fitz.Pixmap(pdf_document, xref)
                    
                    # Convertir a PIL Image
                    if pix.n - pix.alpha < 4:  # GRAY o RGB
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_data))
                        
                        # Convertir a base64
                        buffered = io.BytesIO()
                        pil_image.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        images_data.append({
                            'page_number': page_num + 1,
                            'image_index': img_index,
                            'image_base64': img_base64,
                            'extraction_method': 'embedded'
                        })
                    
                    pix = None
                    
                except Exception as e:
                    print(f"Error extrayendo imagen embebida {img_index} de página {page_num + 1}: {e}")
                    continue
        
        pdf_document.close()
        
        # Método 2: Convertir páginas completas a imágenes (fallback)
        if len(images_data) == 0:
            images_data = await self._convert_pdf_pages_to_images(pdf_data)
        
        return images_data
    
    async def _convert_pdf_pages_to_images(self, pdf_data: bytes) -> List[Dict[str, Any]]:
        """Convierte páginas completas del PDF a imágenes"""
        images_data = []
        
        try:
            # Convertir páginas a imágenes
            images = convert_from_bytes(pdf_data, dpi=200)
            
            for page_num, image in enumerate(images):
                # Convertir a base64
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                images_data.append({
                    'page_number': page_num + 1,
                    'image_index': 0,
                    'image_base64': img_base64,
                    'extraction_method': 'page_conversion'
                })
                
        except Exception as e:
            raise Exception(f"Error convirtiendo páginas a imágenes: {e}")
        
        return images_data
    
    async def _search_relevant_images(
        self, 
        images_data: List[Dict[str, Any]], 
        search_term: str,
        max_results: int
    ) -> List[ImageSearchResult]:
        """Busca imágenes relevantes usando OpenAI Vision API"""
        
        if not images_data:
            return []
        
        # Crear tareas para procesar imágenes en paralelo (con límite)
        semaphore = asyncio.Semaphore(3)  # Limitar a 3 requests simultáneos
        tasks = []
        
        for img_data in images_data:
            task = self._analyze_image_relevance(semaphore, img_data, search_term)
            tasks.append(task)
        
        # Ejecutar análisis en paralelo
        analysis_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar y ordenar resultados
        valid_results = []
        for i, result in enumerate(analysis_results):
            if isinstance(result, Exception):
                print(f"Error analizando imagen {i}: {result}")
                continue
            
            if result and result.confidence > 0.3:  # Umbral de confianza
                valid_results.append(result)
        
        # Ordenar por confianza y limitar resultados
        valid_results.sort(key=lambda x: x.confidence, reverse=True)
        return valid_results[:max_results]
    
    async def _analyze_image_relevance(
        self, 
        semaphore: asyncio.Semaphore,
        img_data: Dict[str, Any], 
        search_term: str
    ) -> Optional[ImageSearchResult]:
        """
        Analiza una imagen para determinar su relevancia al término de búsqueda
        Usa solo herramientas de Python ya que DeepSeek no soporta Vision API
        """
        
        async with semaphore:
            try:
                # Análisis completo con herramientas de Python
                python_analysis = self._analyze_image_with_python_tools(img_data['image_base64'])
                
                # Análisis de relevancia basado en características técnicas
                relevance_score = self._calculate_relevance_score(python_analysis, search_term)
                
                if relevance_score > 0.3:  # Umbral de confianza
                    # Crear descripción basada en análisis técnico
                    description = self._create_description_from_analysis(python_analysis, search_term)
                    
                    return ImageSearchResult(
                        page_number=img_data['page_number'],
                        image_base64=img_data['image_base64'],
                        description=description,
                        confidence=relevance_score
                    )
                
                return None
                
            except Exception as e:
                print(f"Error analizando imagen: {e}")
                return None
    
    def _calculate_relevance_score(self, analysis: Dict[str, Any], search_term: str) -> float:
        """
        Calcula un score de relevancia basado en características técnicas
        """
        try:
            base_score = 0.0
            
            # Score basado en calidad de imagen
            quality_score = analysis.get('quality_score', 0.0)
            if quality_score > 0.7:
                base_score += 0.3
            elif quality_score > 0.5:
                base_score += 0.2
            elif quality_score > 0.3:
                base_score += 0.1
            
            # Score basado en detección de objetos
            objects_detected = analysis.get('objects_detected', 0)
            if objects_detected > 2:
                base_score += 0.3
            elif objects_detected > 0:
                base_score += 0.2
            
            # Score basado en texto detectado
            if analysis.get('text_detected', False):
                base_score += 0.2
            
            # Score basado en colores (productos textiles suelen tener colores variados)
            dominant_colors = analysis.get('dominant_colors', [])
            if len(dominant_colors) > 2:
                base_score += 0.2
            
            # Bonus por búsquedas específicas
            search_term_lower = search_term.lower()
            
            # Heurísticas específicas para términos comunes
            if search_term_lower in ['acolchado', 'colcha', 'edredón', 'ropa de cama']:
                # Para textiles, buscar patrones de color y textura
                if len(dominant_colors) > 1 and objects_detected > 0:
                    base_score += 0.3
            
            elif search_term_lower in ['mueble', 'silla', 'mesa', 'sofá']:
                # Para muebles, buscar formas geométricas
                if objects_detected > 0:
                    base_score += 0.4
            
            elif search_term_lower in ['producto', 'artículo', 'item']:
                # Para productos genéricos, cualquier objeto detectado es relevante
                if objects_detected > 0:
                    base_score += 0.5
            
            # Penalización por muy baja calidad
            if quality_score < 0.2:
                base_score *= 0.5
            
            return min(base_score, 1.0)
            
        except Exception as e:
            print(f"Error calculando relevancia: {e}")
            return 0.0
    
    def _create_description_from_analysis(self, analysis: Dict[str, Any], search_term: str) -> str:
        """
        Crea una descripción basada en el análisis técnico
        """
        try:
            description_parts = []
            
            # Información básica
            dimensions = analysis.get('dimensions', (0, 0))
            description_parts.append(f"Imagen de {dimensions[0]}x{dimensions[1]} píxeles")
            
            # Calidad
            quality_score = analysis.get('quality_score', 0.0)
            if quality_score > 0.8:
                description_parts.append("de alta calidad")
            elif quality_score > 0.6:
                description_parts.append("de buena calidad")
            elif quality_score > 0.4:
                description_parts.append("de calidad media")
            else:
                description_parts.append("de baja calidad")
            
            # Objetos detectados
            objects_detected = analysis.get('objects_detected', 0)
            if objects_detected > 3:
                description_parts.append(f"con múltiples objetos detectados ({objects_detected})")
            elif objects_detected > 1:
                description_parts.append(f"con {objects_detected} objetos detectados")
            elif objects_detected == 1:
                description_parts.append("con un objeto principal")
            
            # Colores dominantes
            dominant_colors = analysis.get('dominant_colors', [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                description_parts.append(f"con colores predominantes: {color_description}")
            
            # Texto
            if analysis.get('text_detected', False):
                description_parts.append("que contiene texto")
            
            # Brillo y contraste
            brightness = analysis.get('brightness', 0)
            contrast = analysis.get('contrast', 0)
            
            if brightness > 200:
                description_parts.append("con iluminación brillante")
            elif brightness < 80:
                description_parts.append("con iluminación tenue")
            
            if contrast > 60:
                description_parts.append("y alto contraste")
            elif contrast < 30:
                description_parts.append("y bajo contraste")
            
            # Relación con el término de búsqueda
            search_term_lower = search_term.lower()
            if search_term_lower in ['acolchado', 'colcha', 'edredón']:
                description_parts.append(f"Posible contenido relacionado con {search_term}")
            elif objects_detected > 0:
                description_parts.append(f"Contenido potencialmente relacionado con {search_term}")
            
            return ". ".join(description_parts).capitalize() + "."
            
        except Exception as e:
            return f"Imagen analizada técnicamente. Posible relación con {search_term}."
    
    def _describe_colors(self, colors: List[List[int]]) -> str:
        """
        Describe los colores de manera amigable
        """
        try:
            color_names = []
            for color in colors[:3]:  # Solo los primeros 3 colores
                r, g, b = color[:3]
                
                # Clasificación básica de colores
                if r > 200 and g > 200 and b > 200:
                    color_names.append("blanco")
                elif r < 50 and g < 50 and b < 50:
                    color_names.append("negro")
                elif r > 200 and g < 100 and b < 100:
                    color_names.append("rojo")
                elif r < 100 and g > 200 and b < 100:
                    color_names.append("verde")
                elif r < 100 and g < 100 and b > 200:
                    color_names.append("azul")
                elif r > 200 and g > 200 and b < 100:
                    color_names.append("amarillo")
                elif r > 200 and g < 200 and b > 200:
                    color_names.append("magenta")
                elif r < 200 and g > 200 and b > 200:
                    color_names.append("cian")
                elif r > 150 and g > 100 and b < 100:
                    color_names.append("marrón")
                elif r > 150 and g > 150 and b > 150:
                    color_names.append("gris claro")
                elif r < 150 and g < 150 and b < 150:
                    color_names.append("gris oscuro")
                else:
                    color_names.append("color mixto")
            
            return ", ".join(color_names)
            
        except Exception:
            return "colores variados"
    
    async def describe_image(self, image_base64: str) -> str:
        """
        Describe una imagen usando análisis técnico con herramientas de Python
        ya que DeepSeek no soporta Vision API
        """
        try:
            # Análisis con herramientas de Python
            python_analysis = self._analyze_image_with_python_tools(image_base64)
            
            # Crear descripción detallada basada en análisis técnico
            description = self._create_detailed_description(python_analysis)
            
            # Usar DeepSeek para mejorar la descripción con context técnico
            enhanced_description = await self._enhance_description_with_ai(description, python_analysis)
            
            return enhanced_description
            
        except Exception as e:
            return f"Error describiendo imagen: {str(e)}"
    
    def _create_detailed_description(self, analysis: Dict[str, Any]) -> str:
        """
        Crea una descripción detallada basada en el análisis técnico
        """
        try:
            parts = []
            
            # Información básica
            dimensions = analysis.get('dimensions', (0, 0))
            parts.append(f"Esta es una imagen de {dimensions[0]} x {dimensions[1]} píxeles")
            
            # Calidad general
            quality_score = analysis.get('quality_score', 0.0)
            if quality_score > 0.8:
                parts.append("de excelente calidad visual")
            elif quality_score > 0.6:
                parts.append("de buena calidad")
            elif quality_score > 0.4:
                parts.append("de calidad media")
            else:
                parts.append("de calidad básica")
            
            # Características técnicas
            brightness = analysis.get('brightness', 0)
            contrast = analysis.get('contrast', 0)
            sharpness = analysis.get('sharpness', 0)
            
            # Brillo
            if brightness > 200:
                parts.append("con iluminación muy brillante")
            elif brightness > 150:
                parts.append("con buena iluminación")
            elif brightness > 100:
                parts.append("con iluminación moderada")
            else:
                parts.append("con iluminación tenue")
            
            # Contraste
            if contrast > 70:
                parts.append("y alto contraste")
            elif contrast > 40:
                parts.append("y contraste moderado")
            else:
                parts.append("y bajo contraste")
            
            # Nitidez
            if sharpness > 1000:
                parts.append("La imagen presenta alta nitidez")
            elif sharpness > 500:
                parts.append("La imagen tiene nitidez media")
            else:
                parts.append("La imagen presenta baja nitidez")
            
            # Contenido detectado
            objects_detected = analysis.get('objects_detected', 0)
            if objects_detected > 5:
                parts.append(f"Se detectaron múltiples objetos ({objects_detected}), sugiriendo una composición compleja")
            elif objects_detected > 2:
                parts.append(f"Se detectaron {objects_detected} objetos principales")
            elif objects_detected > 0:
                parts.append(f"Se detectó {objects_detected} objeto principal")
            else:
                parts.append("No se detectaron objetos claramente definidos")
            
            # Texto
            if analysis.get('text_detected', False):
                parts.append("La imagen contiene texto o elementos textuales")
            
            # Colores
            dominant_colors = analysis.get('dominant_colors', [])
            if len(dominant_colors) > 0:
                color_description = self._describe_colors(dominant_colors)
                parts.append(f"Los colores predominantes son: {color_description}")
            
            # Análisis de composición
            if objects_detected > 0 and quality_score > 0.6:
                parts.append("La composición sugiere que podría ser una fotografía de producto o contenido comercial")
            elif analysis.get('text_detected', False) and objects_detected > 0:
                parts.append("La combinación de texto e imágenes sugiere contenido informativo o publicitario")
            
            return ". ".join(parts) + "."
            
        except Exception as e:
            return f"Análisis técnico de imagen disponible. Error en descripción detallada: {str(e)}"
    
    async def _enhance_description_with_ai(self, technical_description: str, analysis: Dict[str, Any]) -> str:
        """
        Mejora la descripción técnica usando DeepSeek con solo texto
        """
        try:
            prompt = f"""
            Basándote en este análisis técnico de una imagen, crea una descripción más natural y comprensible:

            ANÁLISIS TÉCNICO:
            {technical_description}

            DATOS ESPECÍFICOS:
            - Dimensiones: {analysis.get('dimensions', 'N/A')}
            - Calidad: {analysis.get('quality_score', 0):.2f}/1.0
            - Brillo: {analysis.get('brightness', 0):.1f}/255
            - Contraste: {analysis.get('contrast', 0):.1f}
            - Nitidez: {analysis.get('sharpness', 0):.1f}
            - Objetos detectados: {analysis.get('objects_detected', 0)}
            - Texto detectado: {'Sí' if analysis.get('text_detected', False) else 'No'}

            Instrucciones:
            1. Reescribe la descripción de manera más natural y fluida
            2. Mantén la información técnica pero hazla más comprensible
            3. Sugiere qué tipo de contenido podría ser basándote en las características
            4. Responde en español
            5. Máximo 200 palabras
            """
            
            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250
            )
            
            ai_description = response.choices[0].message.content.strip()
            
            # Combinar descripción AI con datos técnicos
            final_description = f"""{ai_description}

---
ANÁLISIS TÉCNICO:
• Resolución: {analysis.get('dimensions', 'N/A')}
• Calidad: {analysis.get('quality_score', 0):.2f}/1.0
• Brillo: {analysis.get('brightness', 0):.1f}/255
• Contraste: {analysis.get('contrast', 0):.1f}
• Nitidez: {'Alta' if analysis.get('sharpness', 0) > 1000 else 'Media' if analysis.get('sharpness', 0) > 500 else 'Baja'}
• Objetos detectados: {analysis.get('objects_detected', 0)}
• Contiene texto: {'Sí' if analysis.get('text_detected', False) else 'No'}"""
            
            return final_description
            
        except Exception as e:
            # Si falla la mejora con AI, retornar descripción técnica
            return f"{technical_description}\n\n[Error mejorando descripción con AI: {str(e)}]"
    
    async def advanced_image_analysis(self, image_base64: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Análisis avanzado de imagen usando herramientas de Python y DeepSeek (solo texto)
        
        Args:
            image_base64: Imagen en base64
            analysis_type: Tipo de análisis ("general", "product", "document", "quality")
            
        Returns:
            Diccionario con análisis completo
        """
        try:
            # Análisis con herramientas de Python
            python_analysis = self._analyze_image_with_python_tools(image_base64)
            
            # Crear análisis basado en características técnicas
            ai_analysis = await self._create_ai_analysis_from_technical(python_analysis, analysis_type)
            
            # Combinar análisis
            complete_analysis = {
                "ai_analysis": ai_analysis,
                "technical_analysis": python_analysis,
                "combined_quality_score": (
                    ai_analysis.get("confidence", 0.5) + 
                    python_analysis.get("quality_score", 0.5)
                ) / 2,
                "analysis_type": analysis_type
            }
            
            return complete_analysis
            
        except Exception as e:
            return {"error": f"Error en análisis avanzado: {str(e)}"}
    
    async def _create_ai_analysis_from_technical(self, python_analysis: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """
        Crea análisis AI basado en datos técnicos
        """
        try:
            # Crear descripción técnica
            technical_summary = self._create_detailed_description(python_analysis)
            
            # Prompts específicos según el tipo
            type_prompts = {
                "general": "Basándote en este análisis técnico, describe los elementos generales de la imagen",
                "product": "Basándote en este análisis técnico, evalúa si esto podría ser un producto comercial",
                "document": "Basándote en este análisis técnico, evalúa si esto es contenido documental",
                "quality": "Basándote en este análisis técnico, evalúa la calidad general de la imagen"
            }
            
            prompt = f"""
            {type_prompts.get(analysis_type, type_prompts["general"])}
            
            ANÁLISIS TÉCNICO:
            {technical_summary}
            
            DATOS ESPECÍFICOS:
            - Dimensiones: {python_analysis.get('dimensions', 'N/A')}
            - Calidad: {python_analysis.get('quality_score', 0):.2f}/1.0
            - Objetos detectados: {python_analysis.get('objects_detected', 0)}
            - Texto detectado: {'Sí' if python_analysis.get('text_detected', False) else 'No'}
            - Colores dominantes: {len(python_analysis.get('dominant_colors', []))} colores
            
            Responde en JSON con:
            {{
                "summary": "resumen de 1-2 líneas",
                "elements": ["lista", "de", "elementos", "detectados"],
                "colors": ["descripción", "de", "colores"],
                "quality_assessment": "evaluación de calidad",
                "recommendations": ["recomendaciones", "basadas", "en", "análisis"],
                "confidence": 0.0-1.0
            }}
            """
            
            response = await self.client.chat.completions.create(
                model=self.vision_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            
            content = response.choices[0].message.content.strip()
            
            # Intentar parsear JSON
            import json
            try:
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                json_str = content[start_idx:end_idx]
                ai_analysis = json.loads(json_str)
            except:
                # Fallback si no se puede parsear JSON
                ai_analysis = {
                    "summary": "Análisis técnico completado",
                    "elements": ["objetos detectados", "características técnicas"],
                    "colors": [self._describe_colors(python_analysis.get('dominant_colors', []))],
                    "quality_assessment": f"Calidad: {python_analysis.get('quality_score', 0):.2f}/1.0",
                    "recommendations": ["Imagen procesada con herramientas técnicas"],
                    "confidence": python_analysis.get('quality_score', 0.5)
                }
            
            return ai_analysis
            
        except Exception as e:
            return {
                "summary": f"Error en análisis AI: {str(e)}",
                "elements": ["análisis técnico disponible"],
                "colors": ["colores detectados"],
                "quality_assessment": "Análisis técnico completado",
                "recommendations": ["Revisar manualmente"],
                "confidence": 0.3
            }
    
    def enhance_image_quality(self, image_base64: str) -> str:
        """
        Mejora la calidad de una imagen usando herramientas de Python
        
        Args:
            image_base64: Imagen en base64
            
        Returns:
            Imagen mejorada en base64
        """
        try:
            # Decodificar imagen
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Aplicar mejoras
            enhanced = image
            
            # Ajustar brillo y contraste
            enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = enhancer.enhance(1.1)  # Ligeramente más brillante
            
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.2)  # Más contraste
            
            # Mejurar nitidez
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.3)  # Más nitidez
            
            # Aplicar filtro de suavizado si es necesario
            enhanced = enhanced.filter(ImageFilter.SMOOTH_MORE)
            
            # Convertir de vuelta a base64
            buffered = io.BytesIO()
            enhanced.save(buffered, format="PNG")
            enhanced_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return enhanced_base64
            
        except Exception as e:
            print(f"Error mejorando imagen: {e}")
            return image_base64  # Retornar imagen original si hay error