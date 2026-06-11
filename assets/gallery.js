(function () {
  const lang = document.documentElement.lang || "en";
  const isZh = lang.toLowerCase().startsWith("zh");

  const copy = {
    open: isZh ? "点击查看大图" : "Click to open",
    close: isZh ? "关闭" : "Close",
    previous: isZh ? "上一张" : "Previous",
    next: isZh ? "下一张" : "Next",
    t2iHint: isZh ? "点击查看文本提示" : "Click to view the text prompt",
    it2tHint: isZh
      ? "点击查看指令和回答"
      : "Click to view the instruction and answer",
  };

  const getGalleryHint = (carousel) => {
    const firstSlide = carousel.querySelector(".sample-slide");
    if (firstSlide && firstSlide.id.startsWith("it2t-")) {
      return copy.it2tHint;
    }
    return copy.t2iHint;
  };

  const createLightbox = () => {
    const lightbox = document.createElement("div");
    lightbox.className = "sample-lightbox";
    lightbox.setAttribute("role", "dialog");
    lightbox.setAttribute("aria-modal", "true");
    lightbox.setAttribute("aria-hidden", "true");
    lightbox.innerHTML = `
      <button class="sample-lightbox-button sample-lightbox-close" type="button" aria-label="${copy.close}">×</button>
      <button class="sample-lightbox-button sample-lightbox-prev" type="button" aria-label="${copy.previous}">‹</button>
      <figure class="sample-lightbox-figure">
        <img alt="" />
        <figcaption></figcaption>
      </figure>
      <button class="sample-lightbox-button sample-lightbox-next" type="button" aria-label="${copy.next}">›</button>
    `;
    document.body.append(lightbox);
    return lightbox;
  };

  const lightbox = createLightbox();
  const lightboxImage = lightbox.querySelector("img");
  const lightboxCaption = lightbox.querySelector("figcaption");
  let activeSlides = [];
  let activeIndex = 0;

  const visibleSlides = (carousel) =>
    Array.from(carousel.querySelectorAll(".sample-slide")).filter(
      (slide) => !slide.hidden
    );

  const renderLightbox = () => {
    const slide = activeSlides[activeIndex];
    if (!slide) return;

    const img = slide.querySelector("img");
    const caption = slide.querySelector("figcaption");
    lightboxImage.src = img.currentSrc || img.src;
    lightboxImage.alt = img.alt || "";
    lightboxCaption.innerHTML = caption ? caption.innerHTML : "";
  };

  const openLightbox = (carousel, slide) => {
    activeSlides = visibleSlides(carousel);
    activeIndex = Math.max(0, activeSlides.indexOf(slide));
    renderLightbox();
    lightbox.classList.add("is-open");
    lightbox.setAttribute("aria-hidden", "false");
    document.body.classList.add("has-open-lightbox");
  };

  const closeLightbox = () => {
    lightbox.classList.remove("is-open");
    lightbox.setAttribute("aria-hidden", "true");
    document.body.classList.remove("has-open-lightbox");
  };

  const stepLightbox = (direction) => {
    if (!activeSlides.length) return;
    activeIndex =
      (activeIndex + direction + activeSlides.length) % activeSlides.length;
    renderLightbox();
  };

  lightbox
    .querySelector(".sample-lightbox-close")
    .addEventListener("click", closeLightbox);
  lightbox
    .querySelector(".sample-lightbox-prev")
    .addEventListener("click", () => stepLightbox(-1));
  lightbox
    .querySelector(".sample-lightbox-next")
    .addEventListener("click", () => stepLightbox(1));
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) closeLightbox();
  });

  document.addEventListener("keydown", (event) => {
    if (!lightbox.classList.contains("is-open")) return;
    if (event.key === "Escape") closeLightbox();
    if (event.key === "ArrowLeft") stepLightbox(-1);
    if (event.key === "ArrowRight") stepLightbox(1);
  });

  const enhanceCarousel = (carousel, index) => {
    const slides = Array.from(carousel.querySelectorAll(".sample-slide"));
    if (!slides.length) return;

    carousel.classList.add("is-enhanced");

    slides.forEach((slide) => {
      const img = slide.querySelector("img");
      if (!img) return;

      img.setAttribute("loading", "lazy");
      img.setAttribute("decoding", "async");
      img.setAttribute("tabindex", "0");
      img.setAttribute("role", "button");
      img.setAttribute("aria-label", `${copy.open}: ${img.alt || ""}`);

      img.addEventListener("click", () => openLightbox(carousel, slide));
      img.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          openLightbox(carousel, slide);
        }
      });
    });

    const controls = document.createElement("div");
    controls.className = "sample-gallery-controls";
    controls.setAttribute(
      "aria-label",
      isZh ? "样本查看提示" : "Sample viewing hint"
    );

    const hint = document.createElement("p");
    hint.className = "sample-gallery-hint";
    hint.textContent = getGalleryHint(carousel);

    controls.append(hint);
    carousel.prepend(controls);

    carousel.dataset.galleryIndex = String(index + 1);
  };

  document.addEventListener("DOMContentLoaded", () => {
    document
      .querySelectorAll(".sample-carousel")
      .forEach((carousel, index) => enhanceCarousel(carousel, index));
  });
})();
