#let config(body) = {
  set page(
    margin: (x: 2.5cm, y: 2.5cm), //Para trabajos digitales e impresos simples
    // margin: (left: 3.8cm, y: 2.5cm, right: 2.5cm) //Para trabajos impresos a doble cara
  )
  set text(lang: "es")
  set par(justify: true)

  show heading: set text(font: "Roboto")
  set text(font: "Merriweather 48pt")
  show raw: set text(font: "IBM Plex Mono")
  show math.equation: set text(font: "Asana Math")

  show raw.where(block: false): set text(size: 1.2em)
  show math.equation: set block(breakable: true)

  set quote(block: true)
  set heading(numbering: "1.")
  show heading.where(level: 3): set heading(numbering: none)
  show heading.where(level: 3): set heading(outlined: false)
  show heading.where(level: 4): set heading(numbering: none)
  show heading.where(level: 4): set heading(outlined: false)
  show heading.where(level: 5): set heading(numbering: none)
  show heading.where(level: 5): set heading(outlined: false)
  set table(stroke: black, align: center + horizon, inset: 0.7em)
  set grid(stroke: black, align: center + horizon, inset: 0.7em)

  show link: set text(fill: blue)
  show link: underline

  body
}

#let separador = block(line(length: 100%))
