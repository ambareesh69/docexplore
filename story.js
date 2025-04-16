/* globals scrollama, bootstrap, dataLink, charsPerPixel */

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import { html, render } from "./node_modules/lit-html/lit-html.js";
import { repeat } from "./node_modules/lit-html/directives/repeat.js";
import { num, pc1 } from "./node_modules/@gramex/ui/dist/format.js";
import { insightTree, LEVEL, RANK, GROUP, PARENT } from "./node_modules/@gramex/insighttree/dist/insighttree.js";
import { documap } from "./node_modules/@gramex/documap/dist/documap.js";
import { network } from "./node_modules/@gramex/network/dist/network.js";
import { layer } from "./node_modules/@gramex/chartbase/dist/chartbase.js";
import { marked } from "./node_modules/marked/lib/marked.esm.js";

for (const el of document.querySelectorAll("[data-markdown]")) {
  el.innerHTML = marked(el.innerHTML);
  el.removeAttribute("data-markdown");
}

const data = await fetch(dataLink).then((r) => r.json());

data.topics.forEach((topic, i) => (topic.index = i));
data.docs.forEach((doc, i) => (doc.index = i));
data.topicColor = d3.scaleOrdinal(d3.schemeCategory10).domain(data.topics.map((d) => d.index));

// Flatten the column hierarchies into linear hierarchies via sequence().
// Change [ { topic, subtopic1 }, { topic, subtopic2 }, ... ] to [ topic, subtopic1, subtopic2, ... ]
data.topicHierarchy = sequence(data.topics, ["topic", "subtopic"]);
// Change [ { chapter, section, para1 }, { chapter, section, para2 }, ... ] to [ chapter, section, para, para2, ... ]
data.docHierarchy = sequence(data.docs, ["chapter", "section", "para"]);
// Ensure that the matches now reflect the correct order
const topicMap = Object.fromEntries(data.topicHierarchy.map(({ index }, i) => [index, i]));
const docMap = Object.fromEntries(data.docHierarchy.map(({ index }, i) => [index, i]));

const $minSimilarity = document.querySelector("#min-similarity");
$minSimilarity.classList.remove("d-none");
$minSimilarity.addEventListener("input", drawStory);
const similarityTooltip = new bootstrap.Tooltip("#min-similarity");

// Create a link between every pair of topics
const links = [];
const linkLookup = {};

function getLinks() {
  // Initialize linkLookup
  if (Object.keys(linkLookup).length === 0)
    for (let i = 0; i < data.topics.length; i++)
      for (let j = i + 1; j < data.topics.length; j++)
        links.push((linkLookup[`${i}-${j}`] = { source: i, target: j, count: 0 }));

  const topicIndexSet = new Set(data.topics.map((d) => d.index));
  // Loop through each doc. For every pair of topics in the doc, add a link between the topics
  for (let i = 0; i < data.topics.length; i++)
    for (let j = i + 1; j < data.topics.length; j++) linkLookup[`${i}-${j}`].count = 0;
  for (const doc of data.docs) {
    const docTopics = doc.topics.filter((topicId) => topicIndexSet.has(topicId));
    for (let i = 0; i < docTopics.length; i++)
      for (let j = i + 1; j < docTopics.length; j++) linkLookup[`${docTopics[i]}-${docTopics[j]}`].count++;
  }
  return links.filter((d) => d.count > 0);
}

let scroller;

function drawStory() {
  similarityTooltip.setContent({ ".tooltip-inner": `Minimum similarity to match: ${pc1($minSimilarity.value)}` });
  const tree = drawTopics(data);
  const subtopics = {};
  let lastTopic;
  for (const row of tree.tree) {
    if (row[LEVEL] == 1) subtopics[(lastTopic = row[GROUP])] = [];
    if (row[LEVEL] == 2) subtopics[lastTopic].push(row);
  }
  const topics = Object.keys(subtopics);
  const chapters = new Set(data.docs.map((doc) => doc.chapter));
  const sections = new Set(data.docs.map((doc) => doc.section));

  const docStory = html`
      <div class="step p-4" data-panel-target="topics" data-group="">
        <h2>Here are the topics most discussed in the document.</h2>
      </div>
      ${Object.entries(subtopics).map(
        ([topic, subtopics], i) =>
          html`<div class="step p-4" data-panel-target="topics" data-group="${topic}">
            <h2>
              ${i == 0
                ? html`<span style="color:${data.topicColor(topic)}">${topic}</span> was the most discussed`
                : html`... followed by <span style="color:${data.topicColor(topic)}">${topic}</span>`}
            </h2>
            ${subtopics.length > 4 && subtopics[0].count > 2
              ? html`<p class="my-4">
                  Emphasis was on <strong>${subtopics.at(0)[GROUP]}</strong> and
                  <strong>${subtopics.at(1)[GROUP]}</strong> more than <em>${subtopics.at(-1)[GROUP]}</em> or
                  <em>${subtopics.at(-2)[GROUP]}</em>.
                </p>`
              : subtopics.length > 2 && subtopics[0].count > 2
                ? html`<p class="my-4">
                    Emphasis was on <strong>${subtopics.at(0)[GROUP]}</strong> and
                    <strong>${subtopics.at(1)[GROUP]}</strong>.
                  </p>`
                : ""}
          </div>`,
      )}
      <div class="step p-4" data-panel-target="text" data-filter="{}">
        <h2>Explore the document</h2>
        <p>Each of the ${chapters.size} chapters are shown as blue circles
          <svg width="1rem" height="1rem"><circle cx="50%" cy="50%" r="50%" fill="var(--bs-primary)"></svg>
        </p>
        <ol>
          ${[...chapters].slice(0, 3).map((s) => html`<li>${s}</li>`)}
          <li>...</li>
        </ol>
        <p>These are divided into ${sections.size} sections shown as grey circles
          <svg width="1rem" height="1rem"><circle cx="50%" cy="50%" r="50%" fill="#888"></svg>.
        </p>
        <p>Each paragraph is then shown as a grey rectangle. Longer paragraphs are wider.</p>
        <p class="h4">Click to read any paragraph.</p>
      </div>
      ${topics.slice(0, 2).map(
        (topic) => html`
          <div class="step p-4" data-panel-target="text" data-filter="${JSON.stringify({ topics: [topic] })}">
            <h2>Explore <span style="color:${data.topicColor(topic)}">${topic}</span> mentions</h2>
            <p>
              Each topic and subtopic are shown as colored badges on top. Click any badge to see which paragraphs
              mention it.
            </p>
          </div>
        `,
      )}
      <div class="step p-4" data-panel-target="text" data-filter="${JSON.stringify({ topics })}">
        <h2>Explore all topics</h2>
        <p>Each topic and subtopic are shown as colored badges on top. Click any badge to see which paragraphs mention it.</p>
      </div>
      <div class="step p-4" data-panel-target="text" data-filter="${JSON.stringify({ topics })}">
        <h2>Move the slider on top</h2>
        <p>The slider controls the minimum similarity to match a paragraph to a topic.</p>
        <p>Slide <kbd>left</kbd> for <strong>approximate</strong> matches (more results).</p>
        <p>Slide <kbd>right</kbd> for <strong>exact</strong> matches (less results).</p>
      </div>
      <div class="step p-4" data-panel-target="network">
        <h2>Explore topic network</h2>
        <p>Each circle represents a subtopic. The size shows the number of paragraphs that mention it. The color shows the main topic.</p>
        <p>Topics are connected if they are mentioned in the same paragraph.</p>
      </div>
    `;

  // Find the most connected topic and its top connections
  const links = getLinks();
  const mostConnectedIndex = d3.maxIndex(data.topics, (d) => d.count);
  const mostConnected = data.topics[mostConnectedIndex];
  const topConnections = links
    .filter(({ source, target, count }) => (source == mostConnectedIndex || target == mostConnectedIndex) && count > 0)
    .sort((a, b) => b.count - a.count)
    .map(({ source, target }) => (target == mostConnectedIndex ? data.topics[source] : data.topics[target]));
  const networkStory = html`
    <div class="step p-4" data-panel-target="network">
      ${mostConnected.count > 0
        ? html` <h2>${mostConnected.subtopic} is well-connected</h2>
            <p>
              At a ${pc1($minSimilarity.value)} similarity level, ${mostConnected.topic} - ${mostConnected.subtopic} is
              the most connected with other topics. It is often mentioned with:
            </p>
            <ul>
              ${topConnections.slice(0, 3).map((d) => html`<li>${d.topic} - ${d.subtopic}</li>`)}
            </ul>`
        : html`<p>
            At a ${pc1($minSimilarity.value)} similarity level, no topics are mentioned together with other topics.
          </p>`}
    </div>
  `;
  render(
    [
      docStory,
      networkStory,
      html`<div class="step p-4" data-panel-target="network">
        <h2>Craft your own stories</h2>
        <p>Move the slider on top to explore which topics are connected with each other.</p>
      </div> `,
    ],
    document.querySelector(".steps"),
  );

  if (scroller) scroller.destroy();
  scroller = scrollama()
    .setup({ step: ".step" })
    .onStepEnter(drawPanel)
    .onStepExit(({ element }) => element.classList.remove("text-bg-success"));
}

drawStory();

function drawPanel({ element }) {
  document.querySelectorAll(".step.active").forEach((el) => el.classList.remove("active"));
  element.classList.add("active");
  const panel = element.dataset.panelTarget;
  if (panel) {
    // Fade out all panels except the current one
    document.querySelectorAll("[data-panel]").forEach((el) => el.classList.toggle("fade", el.dataset.panel !== panel));
    // If the similarity slider on top was changed, re-draw the ENTIRE story. It affects all panels
    if (element.dataset.similarity && element.dataset.similarity !== $minSimilarity.value) {
      $minSimilarity.value = element.dataset.similarity;
      drawStory();
    }
    // Draw the current panel
    methods[panel]({ ...data, ...element.dataset });
  }
}

function drawTopics({ topics, docs, matches, topicColor, group }) {
  // Re-count the number of matches for each topic and doc, based on the filter
  const minSimilarity = document.querySelector("#min-similarity").value;
  topics.forEach((topic) => (topic.count = 0));
  docs.forEach((doc) => {
    doc.count = 0;
    doc.topics = [];
  });
  matches
    .filter(({ similarity }) => similarity > minSimilarity)
    .forEach(({ doc, topic }) => {
      topics[topic].count++;
      docs[doc].count++;
      docs[doc].topics.push(topic);
    });

  const groups = ["topic", "subtopic"];
  const tree = insightTree("#panel-topics", {
    data: topics,
    groups: groups,
    metrics: ["count"],
    sort: "-count",
    impact: "-count",
    totalGroup: "All Topics",
    render: (el, { tree }) => {
      // Get the max count of tree by level
      const maxCount = d3.rollup(
        tree,
        (v) => d3.max(v, (d) => d.count),
        (d) => d[LEVEL],
      );
      render(
        html`
          ${repeat(
            tree,
            ({ [LEVEL]: level, [GROUP]: group }) => `${level}:${group}`,
            ({ [LEVEL]: level, [RANK]: rank, [GROUP]: group, topic, count }) =>
              html`<div
                data-insight-level="${level}"
                data-insight-rank="${rank}"
                class="text-nowrap my-2"
                style="padding-left:${level * 1}rem"
              >
                <div class="d-flex">
                  <input
                    class="form-check-input me-1"
                    type="checkbox"
                    data-key="${groups[level - 1]}"
                    data-value="${group}"
                    style="background-color:${level >= 1 ? topicColor(topic) : null}"
                  />
                  <span
                    class="pe-2 ${count == 0 ? "text-muted" : ""}"
                    }
                    data-bs-toggle="tooltip"
                    data-bs-placement="right"
                    data-bs-original-title="${num(count)} mentions"
                    >${group}</span
                  >
                  <span class="ms-auto">${num(count)}</span>
                </div>
                <div
                  class="bg-warning"
                  style="margin-left: 1.2rem; width: calc(${(count / maxCount.get(level)) *
                  100}% - 1.2rem); height: 3px"
                ></div>
              </div>`,
          )}
        `,
        el,
      );
    },
  });
  tree.update({ level: 1 });
  if (group) tree.show((d) => d[LEVEL] == 0 || d[GROUP] == group || d[PARENT]?.[GROUP] == group);
  return tree;
}

let documapChart;

function drawText({ filter }) {
  // Only draw markers with minSimilarity from slider
  const minSimilarity = $minSimilarity.value;
  data.docTopicMap = data.matches
    .filter(({ similarity }) => similarity >= minSimilarity)
    .map(({ doc, topic }) => [docMap[doc], topicMap[topic]]);
  // Draw the documap
  documapChart = documap("#panel-text", {
    topics: data.topicHierarchy,
    docs: data.docHierarchy,
    docTopicMap: data.docTopicMap,
    topicLabel: (d) => d.name,
    // Set marker color based on topic
    markerStyle: (toggle) =>
      toggle.attr("r", 3).style("fill", ([, topicId]) => data.topicColor(data.topicHierarchy[topicId].topic)),
    d3,
  });
  // Style the topics and activate the selected ones
  filter = JSON.parse(filter);
  documapChart.topic
    .attr("data-topic-name", (d) => d.topic)
    .style("background-color", (d) => data.topicColor(d.topic))
    .classed("topic-parent", (d) => d.type == "topic")
    .classed("active", (d) => filter.topics?.includes(d.topic))
    .classed("badge text-decoration-none me-2", true);
  // Style the docs
  documapChart.doc
    .attr("width", (d) => d.name.length / charsPerPixel)
    .classed("me-1", true)
    .each(function (d) {
      this.classList.add(`doctype-${d.type}`);
    });
  // Add chapter text inline
  const chapters = documapChart.doc.filter((d) => d.type == "chapter");
  layer(chapters, "text", "chapter")
    .text((d) => d.chapter)
    .attr("x", 10)
    .attr("y", "50%")
    .attr("dy", "0.35em");
  // Add section text inline
  const sections = documapChart.doc.filter((d) => d.type == "section");
  layer(sections, "text", "section")
    .text((d) => (d.section + " ")[0])
    .attr("x", "50%")
    .attr("y", "50%")
    .attr("dy", "0.35em");
  // Disable the click handler. We'll handle it ourselves
  documapChart.topic.on("click.update", null);
  documapChart.update({ topics: (d) => filter.topics?.includes(d.topic) });
}

const $panelText = document.querySelector("#panel-text");
const modal = new bootstrap.Modal(".modal");

$panelText.addEventListener("click", function (event) {
  if (event.target.matches(".documap-topic")) {
    const isActive = event.target.matches(".active");
    const actives = documapChart.topic.nodes().map((node) => node.classList.contains("active"));
    if (event.target.classList.contains("topic-parent")) {
      const topicName = event.target.getAttribute("data-topic-name");
      data.topicHierarchy.forEach((d, i) => (actives[i] = d.topic == topicName ? !isActive : actives[i]));
    } else
      documapChart.topic.nodes().forEach((node, i) => (actives[i] = node == event.target ? !isActive : actives[i]));
    documapChart.update({ topics: (d, i) => actives[i] });
  }

  const docId = event.target.closest("[data-documap-doc]")?.dataset?.documapDoc;
  if (docId) {
    const actives = documapChart.topic.nodes().map((node) => node.classList.contains("active"));
    const doc = data.docHierarchy[+docId];
    const docTopics = data.docTopicMap
      .filter(([_docId]) => _docId == docId)
      .map(([, topicId]) => topicId)
      .filter((topicId) => actives[topicId]);
    render(
      html`
        <h2 class="h5">${doc.chapter}</h2>
        <h3 class="h6">${doc.section}</h3>
        <blockquote>${doc.para}</blockquote>
        ${docTopics.length ? html`<hr />` : null}
        ${repeat(
          docTopics,
          (id) => id,
          (id) =>
            html`<label>
              <input
                class="form-check-input"
                type="checkbox"
                style="background-color:${data.topicColor(data.topicHierarchy[id].topic)}"
              />
              ${data.topicHierarchy[id].topic} - ${data.topicHierarchy[id].subtopic}
            </label>`,
        )}
      `,
      document.querySelector(".modal-body"),
    );
    modal.show();
  }
});

/**
 * Sequences dataframe by columns into a nested structure.
 * @param {object[]} data - The data to sequence.
 * @param {string[]} columns - The columns to sequence by.
 * @returns {object[]} - The sequenced data structure.
 * @description
 *   - Loops through the data and sequences it by the columns into a nested object structure.
 *   - Pushes each sequenced item into a sequencedData array to return the final structure.
 */
function sequence(data, columns) {
  const addedTopics = {};
  const sequencedData = [];
  data.forEach(function (item, index) {
    let parent = addedTopics;
    let hierarchy = {};
    columns.forEach(function (column) {
      hierarchy[column] = item[column];
      if (!parent[item[column]]) {
        parent[item[column]] = {};
        sequencedData.push({ type: column, name: item[column], index, ...hierarchy });
      }
      parent = parent[item[column]];
    });
  });
  return sequencedData;
}

const methods = { topics: drawTopics, text: drawText, network: drawNetwork };

function drawNetwork({ topics, topicColor }) {
  const linksFiltered = getLinks();
  const nodes = topics;
  nodes.forEach((d) => (d.linkCount = 0));
  const maxCount = d3.max(nodes, (d) => d.count);
  const rScale = d3
    .scaleLinear()
    .domain([0, Math.max(maxCount, 10)])
    .range([3, 30]);
  const graph = network("#doc-network", {
    nodes,
    links: linksFiltered,
    forces: { collide: () => d3.forceCollide().radius((d) => rScale(d.count) + 2) },
    d3,
  });
  linksFiltered.forEach((d) => {
    if (d.source.topic != d.target.topic) {
      d.source.linkCount++;
      d.target.linkCount++;
    }
  });
  graph.nodes
    .attr("fill", (d) => topicColor(d.topic))
    .attr("r", (d) => rScale(d.count))
    .attr("data-bs-toggle", "tooltip")
    .attr("title", (d) => `${d.topic} - ${d.subtopic} (${num(d.count)})`)
    .attr("stroke", (d) => (d.linkCount ? "var(--bs-body-color)" : "rgba(var(--bs-body-color-rgb), 0.5)"))
    .attr("stroke-width", (d) => Math.sqrt(d.linkCount) || 0.5);
  graph.links.attr("stroke", "rgba(var(--bs-body-color-rgb), 0.2)").attr("stroke-width", (d) => Math.sqrt(d.count));
}

new bootstrap.Tooltip(".container", { selector: '[data-bs-toggle="tooltip"]' });
