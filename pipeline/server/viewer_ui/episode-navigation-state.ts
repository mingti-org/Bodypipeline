export type EpisodeNavigationStorage = Pick<Storage, "getItem" | "setItem">;

const PAGE_PREFIX = "bp-viewer-episode-page:v1:";
const SCROLL_PREFIX = "bp-viewer-episode-scroll:v1:";

interface StoredEpisodePage {
  episode: number;
  page: number;
}

interface StoredEpisodeScroll {
  episode: number;
  scrollTop: number;
}

function pageKey(repoId: string) {
  return `${PAGE_PREFIX}${repoId}`;
}

function scrollKey(repoId: string, showFlaggedOnly: boolean) {
  return `${SCROLL_PREFIX}${repoId}:${showFlaggedOnly ? "flagged" : "all"}`;
}

function positiveInteger(value: unknown): number | null {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : null;
}

function readStoredEpisodePage(
  storage: EpisodeNavigationStorage | null,
  repoId: string,
): StoredEpisodePage | null {
  if (!storage) return null;
  try {
    const value = JSON.parse(storage.getItem(pageKey(repoId)) || "null");
    const page = positiveInteger(value?.page);
    const episode = Number(value?.episode);
    if (page === null || !Number.isInteger(episode) || episode < 0) return null;
    return { episode, page };
  } catch {
    return null;
  }
}

export function pageForEpisode(
  episodeId: number,
  episodes: readonly number[],
  pageSize: number,
): number | null {
  const index = episodes.indexOf(episodeId);
  return index === -1 ? null : Math.floor(index / pageSize) + 1;
}

export function initialEpisodePage(
  storage: EpisodeNavigationStorage | null,
  repoId: string,
  episodes: readonly number[],
  episodeId: number,
  pageSize: number,
): number {
  const totalPages = Math.max(1, Math.ceil(episodes.length / pageSize));
  const stored = readStoredEpisodePage(storage, repoId);
  if (stored?.episode === episodeId) {
    return Math.min(stored.page, totalPages);
  }
  return pageForEpisode(episodeId, episodes, pageSize) ?? 1;
}

export function writeEpisodePage(
  storage: EpisodeNavigationStorage | null,
  repoId: string,
  page: number,
  episodeId: number,
) {
  if (!storage || positiveInteger(page) === null) return;
  try {
    storage.setItem(
      pageKey(repoId),
      JSON.stringify({ episode: episodeId, page }),
    );
  } catch {
    // Navigation must keep working when browser storage is unavailable.
  }
}

export function readEpisodeScroll(
  storage: EpisodeNavigationStorage | null,
  repoId: string,
  showFlaggedOnly: boolean,
  episodeId: number,
): number | null {
  if (!storage) return null;
  try {
    const value = JSON.parse(
      storage.getItem(scrollKey(repoId, showFlaggedOnly)) || "null",
    ) as StoredEpisodeScroll | null;
    if (value?.episode !== episodeId) return null;
    return Number.isFinite(value.scrollTop) && value.scrollTop >= 0
      ? value.scrollTop
      : null;
  } catch {
    return null;
  }
}

export function writeEpisodeScroll(
  storage: EpisodeNavigationStorage | null,
  repoId: string,
  showFlaggedOnly: boolean,
  scrollTop: number,
  episodeId: number,
) {
  if (!storage || !Number.isFinite(scrollTop) || scrollTop < 0) return;
  try {
    storage.setItem(
      scrollKey(repoId, showFlaggedOnly),
      JSON.stringify({ episode: episodeId, scrollTop: Math.round(scrollTop) }),
    );
  } catch {
    // Navigation must keep working when browser storage is unavailable.
  }
}
