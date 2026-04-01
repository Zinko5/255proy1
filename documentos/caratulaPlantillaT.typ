#let fechaEntrega = datetime(year: 2025, month: 1, day: 1)


#let overlay(img, color, width: auto, height: auto) = layout(bounds => {
  let size = measure(img, width: width, height: height)
  img
  place(top + left, rect(width: 100%, height: 100%, fill: color))
})

#let formatFecha(fecha) = {
  let meses = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
  )
  let dia = str(fecha.day()) // Convertimos el día a cadena
  let mes = meses.at(fecha.month() - 1)
  let año = str(fecha.year()) // Convertimos el año a cadena
  dia + " de " + mes + " del " + año
}

#let caratula(tituloTarea, materia, nombreDocente, estudiantes, fecha) = page(
  background: overlay(image("umsaLogo.png", width: 25em), white.transparentize(30%)),
  margin: (x: 3cm),
  paper: "a4",
)[
  #align(center)[
    #text(size: 16pt)[#pad(top: 0pt)[
      // #text(size: 16pt, stroke: 0.1mm + white)[#pad(top: 100pt)[
      Universidad Mayor de San Andrés\
      Facultad de Ciencias Puras y Naturales\
      Carrera de Informática
    ]]
    #text(size: 30pt)[#pad(top: -5pt)[
      *#tituloTarea*
    ]]
    #text(size: 16pt)[#pad(top: -15pt)[
      #materia
    ]]
  ]
  #pad(top: 260pt)[
    #text(size: 16pt)[
      *Docente:*
    ]
    #text(size: 14pt)[ #pad(left: 20pt, top: -5pt)[
      #nombreDocente
    ]]
  ]
  #pad(top: 15pt)[
    #let lista_estudiantes = if type(estudiantes) == array {
      estudiantes
    } else {
      (estudiantes,) // convierte string en array de un elemento
    }

    #text(size: 16pt)[
      *#if lista_estudiantes.len() == 1 [Estudiante:] else [Estudiantes:]*
    ]
    #text(size: 14pt)[ #pad(left: 20pt, top: -5pt)[
      #list(marker: [•], ..lista_estudiantes)
    ]]
  ]
  #pad(top: 15pt)[
    #text(size: 16pt)[
      *Fecha de entrega:*
    ]
    #text(size: 14pt)[ #pad(left: 20pt, top: -5pt)[
      #formatFecha(fecha)
      // #fechaEntrega.display()
    ]]
  ]
  #place(bottom + center)[
    #text(size: 13pt)[
      *La Paz - Bolivia*
    ]
  ]
]

#caratula(
  "titulo de la tarea",
  "sigla - nombre de la materia",
  "titulo + nombre del docente",
  ("Estudiante 1", "Estudiante 2"),
  datetime(year: 2025, month: 1, day: 1),
)
