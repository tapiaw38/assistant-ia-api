from typing import List, Dict, Any, Optional
from src.core.platform.fileprocessor.processor import ImageSearchResult

def format_images_to_html(
    images: List[ImageSearchResult], 
    max_width: int = 300,
    include_description: bool = True,
    include_confidence: bool = True
) -> str:
    if not images:
        return ""

    html_parts = []

    for image in images:
        img_tag = f'<img src="data:image/png;base64,{image.image_base64}" style="max-width:{max_width}px" alt="Página {image.page_number} - {image.description}"/>'
        html_parts.append(img_tag)

        if include_description or include_confidence:
            description_parts = []

            description_parts.append(f"<strong>Página {image.page_number}</strong>")

            if include_description:
                description_parts.append(image.description)

            if include_confidence:
                description_parts.append(f"(Confianza: {image.confidence:.2f})")

            description_html = f"<p>{': '.join(description_parts)}</p>"
            html_parts.append(description_html)

        html_parts.append("")

    return "\n".join(html_parts)

def format_single_image_to_html(
    image_base64: str,
    page_number: Optional[int] = None,
    description: Optional[str] = None,
    confidence: Optional[float] = None,
    max_width: int = 250
) -> str:
    alt_parts = []
    if page_number is not None:
        alt_parts.append(f"Página {page_number}")
    if description:
        alt_parts.append(description)

    alt_text = " - ".join(alt_parts) if alt_parts else "Imagen"

    img_tag = f'<img src="data:image/png;base64,{image_base64}" style="max-width:{max_width}px" alt="{alt_text}"/>'

    if page_number is not None or description or confidence is not None:
        description_parts = []

        if page_number is not None:
            description_parts.append(f"<strong>Página {page_number}</strong>")

        if description:
            description_parts.append(description)

        if confidence is not None:
            description_parts.append(f"(Confianza: {confidence:.2f})")

        description_html = f"<p>{': '.join(description_parts)}</p>"

        return f"{img_tag}\n{description_html}"

    return img_tag

def format_images_to_markdown(
    images: List[ImageSearchResult],
    include_base64: bool = False
) -> str:
    if not images:
        return ""

    markdown_parts = []

    for i, image in enumerate(images, 1):
        markdown_parts.append(f"## Imagen {i} - Página {image.page_number}")
        markdown_parts.append(f"**Descripción**: {image.description}")
        markdown_parts.append(f"**Confianza**: {image.confidence:.2f}")

        if include_base64:
            img_html = f'<img src="data:image/png;base64,{image.image_base64}" style="max-width:300px" alt="Página {image.page_number}"/>'
            markdown_parts.append(f"\n{img_html}\n")

        markdown_parts.append("---")

    return "\n\n".join(markdown_parts)

def create_image_gallery_html(
    images: List[ImageSearchResult],
    gallery_title: str = "Imágenes Encontradas",
    max_width: int = 300,
    columns: int = 2
) -> str:
    if not images:
        return f"<h3>{gallery_title}</h3><p>No se encontraron imágenes.</p>"

    html_parts = [f"<h3>{gallery_title}</h3>"]
    html_parts.append('<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">')

    for image in images:
        card_html = f'''
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: #f9f9f9;">
            <img src="data:image/png;base64,{image.image_base64}" 
                 style="max-width: 100%; width: {max_width}px; height: auto; border-radius: 4px;" 
                 alt="Página {image.page_number}"/>
            <h4 style="margin: 10px 0 5px 0; color: #333;">Página {image.page_number}</h4>
            <p style="margin: 5px 0; color: #666; font-size: 14px;">{image.description}</p>
            <p style="margin: 5px 0; color: #888; font-size: 12px;">Confianza: {image.confidence:.2f}</p>
        </div>
        '''
        html_parts.append(card_html)

    html_parts.append("</div>")

    return "\n".join(html_parts)

def format_images_for_chat_response(
    images: List[ImageSearchResult],
    search_term: str,
    max_images_to_show: int = 3,
    max_width: Optional[int] = 250
) -> str:
    if not images:
        return f"No se encontraron imágenes relacionadas con '{search_term}'."

    images_to_show = images[:max_images_to_show]

    response_parts = [f"Encontré {len(images)} imagen(es) relacionada(s) con tu busqueda:"]

    for image in images_to_show:
        img_html = f'<img src="data:image/png;base64,{image.image_base64}" style="max-width:{max_width}px" alt="Página {image.page_number} - {image.description}"/>'
        response_parts.append(img_html)
        #response_parts.append(f"**Página {image.page_number}**: {image.description} (Confianza: {image.confidence:.2f})")
        response_parts.append("")  # Línea vacía

    if len(images) > max_images_to_show:
        response_parts.append(f"... y {len(images) - max_images_to_show} imagen(es) más.")

    return "\n".join(response_parts)
