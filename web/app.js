(() => {
  "use strict";

  const data = window.CFB_SCOREBOARD_DATA;
  const $ = (id) => document.getElementById(id);
  const state = { managerSlot: 9, view: "roster" };

  function formatNumber(value, digits = 0) {
    return Number(value || 0).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
  }

  function formatRecord(manager) {
    const tie = manager.ties ? `-${manager.ties}` : "";
    return `${manager.wins}-${manager.losses}${tie}`;
  }

  function initials(name) {
    return name.split(/\s+/).slice(0, 2).map((word) => word[0] || "").join("").toUpperCase();
  }

  function nextGameText(game) {
    if (!game) return "Schedule complete";
    const dateText = new Intl.DateTimeFormat(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(new Date(game.date));
    return `${game.neutral ? "vs" : game.site} ${game.opponent} · ${dateText}`;
  }

  function selectedManager() {
    return data.managers.find((manager) => manager.slot === state.managerSlot) || data.managers[0];
  }

  function makeLogo(team) {
    const wrap = document.createElement("div");
    const image = document.createElement("img");
    image.className = "team-logo";
    image.src = team.logo;
    image.alt = "";
    image.loading = "lazy";
    const fallback = document.createElement("span");
    fallback.className = "logo-fallback";
    fallback.textContent = initials(team.team);
    image.addEventListener("error", () => {
      image.style.display = "none";
      fallback.style.display = "grid";
    });
    wrap.append(image, fallback);
    return wrap;
  }

  function renderRoster() {
    const manager = selectedManager();
    $("managerName").textContent = manager.name;
    $("managerSlot").textContent = manager.slot;
    $("managerRank").textContent = manager.rank;
    $("rosterStatus").textContent = `${manager.rosterCount} teams · draft slot ${manager.slot}`;
    $("fantasyPoints").textContent = formatNumber(manager.fantasyPoints);
    $("combinedRecord").textContent = formatRecord(manager);
    $("projectedEv").textContent = formatNumber(manager.projectedExpectedPoints, 1);
    $("rosterCount").textContent = `${manager.rosterCount} / ${data.teamsPerManager}`;
    $("rosterHeading").textContent = `${manager.name}'s teams`;
    $("rosterPointTotal").textContent = `${formatNumber(manager.fantasyPoints)} PTS`;

    const roster = $("teamRoster");
    roster.replaceChildren();
    if (!manager.teams.length) {
      const empty = document.createElement("p");
      empty.className = "empty-roster";
      empty.textContent = "No teams recorded for this roster.";
      roster.append(empty);
    }
    manager.teams
      .slice()
      .sort((a, b) => b.fantasyPoints - a.fantasyPoints || a.pickNumber - b.pickNumber)
      .forEach((team) => {
        const card = document.createElement("article");
        card.className = "team-card";
        const copy = document.createElement("div");
        copy.className = "team-copy";
        const name = document.createElement("h3");
        name.className = "team-name";
        name.textContent = team.team;
        const meta = document.createElement("p");
        meta.className = "team-meta";
        meta.textContent = `${team.conference} · Rd ${team.round}, Pick ${team.pickNumber}`;
        const next = document.createElement("p");
        next.className = "next-game";
        next.textContent = nextGameText(team.nextGame);
        copy.append(name, meta, next);

        const score = document.createElement("div");
        score.className = "team-score";
        const points = document.createElement("strong");
        points.textContent = formatNumber(team.fantasyPoints);
        const label = document.createElement("span");
        label.textContent = "points";
        const record = document.createElement("small");
        record.textContent = `${team.wins}-${team.losses}${team.ties ? `-${team.ties}` : ""}`;
        score.append(points, label, record);
        card.append(makeLogo(team), copy, score);
        roster.append(card);
      });

    const leaders = $("leaderPreview");
    leaders.replaceChildren();
    data.managers.slice(0, 5).forEach((row) => {
      const item = document.createElement("li");
      item.className = `leader-row${row.slot === manager.slot ? " selected" : ""}`;
      const rank = document.createElement("span");
      rank.className = "leader-rank";
      rank.textContent = row.rank;
      const name = document.createElement("span");
      name.className = "leader-name";
      name.textContent = row.name;
      const points = document.createElement("span");
      points.className = "leader-points";
      points.textContent = `${formatNumber(row.fantasyPoints)} pts`;
      item.append(rank, name, points);
      leaders.append(item);
    });
  }

  function renderStandings() {
    const body = $("standingsBody");
    body.replaceChildren();
    data.managers.forEach((manager) => {
      const row = document.createElement("tr");
      if (manager.slot === state.managerSlot) row.className = "selected";
      const rank = document.createElement("td");
      rank.textContent = manager.rank;
      const managerCell = document.createElement("td");
      const managerWrap = document.createElement("div");
      managerWrap.className = "standing-manager";
      const slot = document.createElement("span");
      slot.className = "slot-badge";
      slot.textContent = manager.slot;
      const name = document.createElement("span");
      name.textContent = manager.name;
      managerWrap.append(slot, name);
      managerCell.append(managerWrap);
      const points = document.createElement("td");
      points.className = "standing-points";
      points.textContent = formatNumber(manager.fantasyPoints);
      const record = document.createElement("td");
      record.textContent = formatRecord(manager);
      const count = document.createElement("td");
      count.textContent = `${manager.rosterCount} / ${data.teamsPerManager}`;
      const ev = document.createElement("td");
      ev.textContent = formatNumber(manager.projectedExpectedPoints, 1);
      row.append(rank, managerCell, points, record, count, ev);
      body.append(row);
    });
  }

  function renderTeamEv() {
    const teams = data.managers
      .flatMap((manager) => manager.teams)
      .sort((a, b) => b.projectedExpectedPoints - a.projectedExpectedPoints || a.projectedRank - b.projectedRank);
    const body = $("teamEvBody");
    body.replaceChildren();
    teams.forEach((team, index) => {
      const row = document.createElement("tr");
      if (team.managerSlot === state.managerSlot) row.className = "selected";
      const rank = document.createElement("td");
      rank.textContent = String(index + 1);
      const teamCell = document.createElement("td");
      const teamWrap = document.createElement("div");
      teamWrap.className = "ev-team";
      const logo = document.createElement("img");
      logo.src = team.logo;
      logo.alt = "";
      logo.loading = "lazy";
      const name = document.createElement("span");
      name.textContent = team.team;
      teamWrap.append(logo, name);
      teamCell.append(teamWrap);
      const owner = document.createElement("td");
      owner.textContent = team.manager;
      const total = document.createElement("td");
      total.className = "ev-total";
      total.textContent = formatNumber(team.projectedExpectedPoints, 2);
      const regular = document.createElement("td");
      regular.className = "ev-component";
      regular.textContent = formatNumber(team.expectedRegularPoints, 2);
      const conference = document.createElement("td");
      conference.className = "ev-component";
      conference.textContent = formatNumber(team.expectedConferencePoints, 2);
      const playoff = document.createElement("td");
      playoff.className = "ev-component";
      playoff.textContent = formatNumber(team.expectedPlayoffPoints, 2);
      const playoffChance = document.createElement("td");
      playoffChance.textContent = `${formatNumber(team.playoffProbability * 100, 1)}%`;
      const current = document.createElement("td");
      current.textContent = formatNumber(team.fantasyPoints);
      row.append(rank, teamCell, owner, total, regular, conference, playoff, playoffChance, current);
      body.append(row);
    });
    $("teamEvCount").textContent = `${teams.length} drafted teams · live model`;
  }

  function renderDraftBoard() {
    const board = $("draftBoard");
    board.replaceChildren();
    for (let round = 1; round <= data.teamsPerManager; round += 1) {
      const section = document.createElement("section");
      section.className = "draft-round";
      const heading = document.createElement("h2");
      heading.textContent = `Round ${round}`;
      section.append(heading);
      data.draftBoard.filter((pick) => pick.round === round).forEach((pick) => {
        const row = document.createElement("div");
        row.className = `draft-pick${pick.managerSlot === state.managerSlot ? " selected" : ""}`;
        const number = document.createElement("span");
        number.className = "pick-number";
        number.textContent = `#${pick.pickNumber}`;
        const manager = document.createElement("span");
        manager.className = "pick-manager";
        manager.textContent = pick.manager;
        const team = document.createElement("span");
        team.className = `pick-team${pick.team ? "" : " open"}`;
        team.textContent = pick.team || "Open pick";
        row.append(number, manager, team);
        section.append(row);
      });
      board.append(section);
    }
  }

  function setView(view, updateHash = true) {
    state.view = ["roster", "standings", "teamEv", "draft"].includes(view) ? view : "roster";
    document.querySelectorAll(".page-view").forEach((node) => { node.hidden = true; });
    $(`${state.view}View`).hidden = false;
    document.querySelectorAll(".view-tab").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === state.view);
    });
    if (updateHash) history.replaceState(null, "", `${location.pathname}${location.search}#${state.view}`);
  }

  function setManager(slot) {
    const parsed = Number(slot);
    if (!data.managers.some((manager) => manager.slot === parsed)) return;
    state.managerSlot = parsed;
    $("managerSelect").value = String(parsed);
    const url = new URL(location.href);
    url.searchParams.set("manager", String(parsed));
    history.replaceState(null, "", `${url.pathname}${url.search}${location.hash}`);
    renderRoster();
    renderStandings();
    renderTeamEv();
    renderDraftBoard();
  }

  function boot() {
    if (!data || !Array.isArray(data.managers)) {
      $("loading").querySelector("p").textContent = "Scoreboard data is unavailable.";
      return;
    }
    const selector = $("managerSelect");
    data.managers.slice().sort((a, b) => a.slot - b.slot).forEach((manager) => {
      const option = document.createElement("option");
      option.value = manager.slot;
      option.textContent = `${manager.slot}. ${manager.name}`;
      selector.append(option);
    });
    const requested = Number(new URL(location.href).searchParams.get("manager"));
    state.managerSlot = data.managers.some((manager) => manager.slot === requested) ? requested : 9;
    selector.value = String(state.managerSlot);
    selector.addEventListener("change", (event) => setManager(event.target.value));
    document.querySelectorAll(".view-tab").forEach((button) => {
      button.addEventListener("click", () => setView(button.dataset.view));
    });
    window.addEventListener("hashchange", () => setView(location.hash.slice(1), false));

    const updated = new Date(data.generatedAt);
    const updatedText = new Intl.DateTimeFormat(undefined, {
      month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short",
    }).format(updated);
    $("seasonLabel").textContent = `${data.season} season`;
    $("footerSeason").textContent = `${data.season} College Football Fantasy`;
    $("lastUpdated").textContent = `Scores refreshed ${updatedText}`;
    $("standingsStamp").textContent = `Updated ${updatedText}`;
    $("draftProgress").textContent = `${data.recordedPicks} of ${data.totalPicks} picks recorded`;
    renderRoster();
    renderStandings();
    renderTeamEv();
    renderDraftBoard();
    setView(location.hash.slice(1) || "roster", false);
    $("loading").hidden = true;
    $("app").hidden = false;
    if (window.lucide) window.lucide.createIcons();
  }

  window.addEventListener("DOMContentLoaded", boot);
})();
