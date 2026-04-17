# DeepPrivacy Demo

A relationship-aware face anonymization framework for social media, built with **SimSwap** and a web-based interface.
This demo allows users to apply **different levels of facial anonymization depending on social relationship context** (e.g., self, close friends, acquaintances, strangers) and **personal privacy preferences**.

This repository contains the demo system for the framework on customizable facial privacy for SNS environments.
---

## Overview

People often want different privacy protections for different viewers on social media.
For example, a user may want:

- minimal modification for close friends,
- moderate anonymization for acquaintances,
- stronger anonymization for strangers or potentially harmful viewers.

This project implements a **selective face anonymization pipeline** that supports:

- **relationship-dependent anonymization**
- **user-selectable privacy levels**
- **web-based preview and interaction**
- **face transformation using SimSwap-based anonymization components**

Rather than using a one-size-fits-all privacy mechanism, the framework is designed to support **socially adaptive privacy control** in SNS-like environments.

---

## Key Features

- **Relationship-aware anonymization**
  - Apply different transformation strengths depending on the viewer category or social relationship.

- **Preference-based privacy customization**
  - Users can choose how strongly their face should be transformed.

- **SNS-style demo interface**
  - Includes a React-based frontend for interactive testing in a social-media-like environment.

- **Backend anonymization pipeline**
  - Flask-based server that handles face processing and transformation.

- **Preview-oriented workflow**
  - Designed for demonstration, prototyping, and user-study scenarios where people compare privacy settings and transformed outputs.

---

## Repository Structure

The repository is organized into a frontend and a backend. The current structure includes a React-based web application and a Flask-based backend server.  [oai_citation:2‡GitHub](https://github.com/dxlabskku/deepprivacy-demo)

```text
.
├── backend
│   ├── arcface_model
│   ├── checkpoints
│   ├── crop_224
│   ├── data
│   ├── demo_file
│   ├── docs
│   ├── insightface_func
│   ├── main.py
│   ├── models
│   ├── options
│   ├── output
│   ├── parsing_model
│   ├── people
│   ├── pg_modules
│   ├── simswaplogo
│   └── util
├── React_instagram_clone
│   ├── public
│   ├── src
│   ├── package.json
│   └── ...
├── README.md
└── requirements.txt
