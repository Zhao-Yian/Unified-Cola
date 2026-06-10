(function () {
  const lang = document.documentElement.lang || "en";
  const isZh = lang.toLowerCase().startsWith("zh");

  const copy = {
    landscape: isZh ? "横图" : "Landscape",
    portrait: isZh ? "竖图" : "Portrait",
    open: isZh ? "点击查看大图" : "Click to open",
    close: isZh ? "关闭" : "Close",
    previous: isZh ? "上一张" : "Previous",
    next: isZh ? "下一张" : "Next",
    hint: isZh
      ? "切换横图/竖图缩略图，点击图片可放大查看。"
      : "Switch between landscape and portrait thumbnails, then click an image to inspect it.",
  };

  const ratioLabels = {
    landscape: copy.landscape,
    portrait: copy.portrait,
  };

  const classify = (img) => {
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) return "landscape";

    const ratio = width / height;
    return ratio >= 1 ? "landscape" : "portrait";
  };

  const readyImage = (img) => {
    if (img.complete && img.naturalWidth) return Promise.resolve();
    return new Promise((resolve) => {
      img.addEventListener("load", resolve, { once: true });
      img.addEventListener("error", resolve, { once: true });
    });
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

  const enhanceCarousel = async (carousel, index) => {
    const slides = Array.from(carousel.querySelectorAll(".sample-slide"));
    if (!slides.length) return;

    carousel.classList.add("is-enhanced");

    await Promise.all(
      slides.map(async (slide) => {
        const img = slide.querySelector("img");
        if (!img) return;

        await readyImage(img);
        const ratio = classify(img);
        slide.dataset.ratio = ratio;

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
      })
    );

    const counts = slides.reduce(
      (acc, slide) => {
        const ratio = slide.dataset.ratio || "landscape";
        acc[ratio] += 1;
        return acc;
      },
      { landscape: 0, portrait: 0 }
    );

    const defaultRatio = counts.landscape ? "landscape" : "portrait";
    carousel.dataset.activeRatio = defaultRatio;

    const controls = document.createElement("div");
    controls.className = "sample-gallery-controls";
    controls.setAttribute(
      "aria-label",
      isZh ? "按宽高比筛选样本" : "Filter samples by aspect ratio"
    );

    const hint = document.createElement("p");
    hint.className = "sample-gallery-hint";
    hint.textContent = copy.hint;

    const chips = document.createElement("div");
    chips.className = "sample-ratio-chips";

    const showRatio = (ratio, shouldScroll = true) => {
      carousel.dataset.activeRatio = ratio;
      chips.querySelectorAll(".sample-ratio-chip").forEach((chip) => {
        const active = chip.dataset.ratio === ratio;
        chip.classList.toggle("is-active", active);
        chip.setAttribute("aria-pressed", active ? "true" : "false");
      });

      slides.forEach((slide) => {
        slide.hidden = slide.dataset.ratio !== ratio;
      });

      const firstVisible = visibleSlides(carousel)[0];
      if (shouldScroll && firstVisible) {
        firstVisible.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
          inline: "start",
        });
      }
    };

    Object.keys(ratioLabels).forEach((ratio) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-ratio-chip";
      button.dataset.ratio = ratio;
      button.textContent = `${ratioLabels[ratio]} ${counts[ratio]}`;
      button.setAttribute(
        "aria-pressed",
        ratio === defaultRatio ? "true" : "false"
      );
      if (ratio === defaultRatio) button.classList.add("is-active");
      if (!counts[ratio]) button.disabled = true;

      button.addEventListener("click", () => showRatio(ratio));

      chips.append(button);
    });

    controls.append(hint, chips);
    carousel.prepend(controls);
    showRatio(defaultRatio, false);

    carousel.dataset.galleryIndex = String(index + 1);
  };

  document.addEventListener("DOMContentLoaded", () => {
    document
      .querySelectorAll(".sample-carousel")
      .forEach((carousel, index) => enhanceCarousel(carousel, index));
  });
})();
