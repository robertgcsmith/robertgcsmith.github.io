(() => {
  const page = document.querySelector("[data-article-filter]");
  if (!page) return;

  const input = page.querySelector("#article-query");
  const clear = page.querySelector(".article-search-clear");
  const topics = page.querySelector(".topic-filters");
  const buttons = topics ? [...topics.querySelectorAll("button")] : [];
  const articles = [...page.querySelectorAll(".archive-item")].map(element => {
    const tags = JSON.parse(element.dataset.topics || "[]") || [];
    return {
      element,
      tags,
      text: `${element.dataset.search} ${tags.join(" ")}`.toLowerCase()
    };
  });
  const count = page.querySelector("[data-article-count]");
  const empty = page.querySelector("[data-article-empty]");

  function filterArticles() {
    const words = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    let slug = "";
    if (topics) {
      try {
        slug = decodeURIComponent(window.location.hash.slice(1));
      } catch {
        slug = window.location.hash.slice(1);
      }
    }
    const selected = buttons.find(button => button.dataset.topicSlug === slug);
    let visible = 0;

    articles.forEach(article => {
      const matchesTopic = slug === "" || (selected && article.tags.includes(selected.dataset.topic));
      const matches = matchesTopic && words.every(word => article.text.includes(word));
      article.element.hidden = !matches;
      if (matches) visible++;
    });
    buttons.forEach(button => {
      button.setAttribute("aria-pressed", String(button === selected));
    });
    const message = `${visible} article${visible === 1 ? "" : "s"}`;
    if (count.textContent !== message) count.textContent = message;
    empty.hidden = visible > 0;
    clear.hidden = input.value.length === 0;
  }

  input.addEventListener("input", filterArticles);
  clear.addEventListener("click", () => {
    input.value = "";
    filterArticles();
    input.focus();
  });
  if (topics) {
    topics.addEventListener("click", event => {
      const button = event.target.closest("button[data-topic-slug]");
      if (!button) return;
      const url = new URL(window.location.href);
      url.hash = button.dataset.topicSlug ? encodeURIComponent(button.dataset.topicSlug) : "";
      if (url.href !== window.location.href) history.pushState(null, "", url);
      filterArticles();
    });
    window.addEventListener("popstate", filterArticles);
    window.addEventListener("hashchange", filterArticles);
  }
  page.querySelector(".archive-controls").hidden = false;
  filterArticles();
})();
