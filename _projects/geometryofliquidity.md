---
layout: page
title: Quantifying the Geometry of Liquidity
description: An end-to-end system for intraday equity markets
img: assets/img/initial_system_architecture.png
importance: 1
category: Quantitative Finance
---
This project has two distinct parts: mathematical research and modelling, and the design of a complete end-to-end algorithmic trading system.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/initial_system_architecture.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Initial project architecture.
</div>

The quantitative research component aims to investigate whether regions of concentrated liquidity and asymmetric buying/selling pressure can be quantified, systematically modelled, and rigorously evaluated to support intraday algorithmic trading decisions in equity markets. The project is based on original mathematical and statistical research.

The system design component functions as an implementation of the mathematical research. Combining market structure principles, volume and participation parametrisations, order-flow imbalance, liquidity response, and zone-boundary behaviour, the system is designed to support the full lifecycle from data ingestion, demand zone detection, signal validation, and trade execution (via novel API-driven architecture).
