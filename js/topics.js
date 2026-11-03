(() => {
  const page = document.querySelector(".topics-page");
  if (!page) return;

  const controls = page.querySelector(".topic-filters");
  const buttons = [...controls.querySelectorAll("button")];
  const articles = [...page.querySelectorAll(".archive-item")].map(element => ({
    element,
    topics: JSON.parse(element.dataset.topics)
  }));
  const count = page.querySelector(".topics-count");
  const empty = page.querySelector(".topics-empty");
  let lastHash;

  function filterArticles() {
    if (lastHash === window.location.hash) return;
    lastHash = window.location.hash;
    let slug;
    try {
      slug = decodeURIComponent(lastHash.slice(1));
    } catch {
      slug = lastHash.slice(1);
    }
    const selected = buttons.find(button => button.dataset.topicSlug === slug);
    let visible = 0;

    articles.forEach(article => {
      const matches = slug === "" || (selected && article.topics.includes(selected.dataset.topic));
      article.element.hidden = !matches;
      if (matches) visible++;
    });
    buttons.forEach(button => {
      button.setAttribute("aria-pressed", String(button === selected));
    });
    count.textContent = `${visible} article${visible === 1 ? "" : "s"}`;
    empty.hidden = visible > 0;
  }

  controls.addEventListener("click", event => {
    const button = event.target.closest("button[data-topic-slug]");
    if (!button) return;
    const url = new URL(window.location.href);
    url.hash = button.dataset.topicSlug ? encodeURIComponent(button.dataset.topicSlug) : "";
    if (url.href !== window.location.href) history.pushState(null, "", url);
    filterArticles();
  });
  window.addEventListener("popstate", filterArticles);
  window.addEventListener("hashchange", filterArticles);
  controls.hidden = false;
  filterArticles();
})();
