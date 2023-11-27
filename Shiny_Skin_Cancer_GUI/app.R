#library(shiny)
#library(shinydashboard,  warn.conflicts = FALSE)
#library(shinyWidgets,  warn.conflicts = FALSE)
#library(bslib)
#library(DT)
#library(dplyr)
#library(stringr)
#library(ggplot2)
#library(torch)
#library(magick)

packages <- c("shiny","shinydashboard","shinyWidgets","bslib","DT","dplyr","stringr","ggplot2","torch","magick", "torchvision", "png")
### WARNING, ACCEPT EXTRA DATA FOR TORCH PACKAGE INCLUDE ###
for(package in packages){
  if (!require(package, character.only = TRUE)) {
    install.packages(package)
    library(package, character.only = TRUE,warn.conflicts = FALSE)
  } else {
    library(package, character.only = TRUE,warn.conflicts = FALSE)
  }
}


# los datos y el codigo están en la carpeta "auxiliar"
#load("auxiliar/df.RData")
#source("auxiliar/graficaydatos.R")

#Aquí por default le está aplicando a los datos una función
#nombresPaises <- leeDatos(df)


#---------- Metadata CSV Load --------------
#Get metadata 
original_metadata <- read.csv('data_source/HAM10000_metadata.csv');
#Quit NA values and sex type "unknown"
original_metadata <- original_metadata[original_metadata$sex!="unknown", ]
original_metadata <- original_metadata[!is.na(original_metadata$age), ]
metadata <- original_metadata[, !(names(original_metadata) %in% c("lesion_id", "image_id"))]

#Change column names
#names(metadata)[names(metadata) == "dx"] <- "Diagnostico"
#names(metadata)[names(metadata) == "dx_type"] <- "Metodo de diagnostico"
#names(metadata)[names(metadata) == "age"] <- "Edad"
#names(metadata)[names(metadata) == "sex"] <- "Sexo"
#names(metadata)[names(metadata) == "localization"] <- "Localizacion"

#-------------------------------------------

#------ Cargar el modelo entrnado ----------

modelo_cargado <- torch_load("data_source/Skin_Cancer_Model.pth")
modelo_cargado$eval()

compose_substitute <- function(x) {
  mean_RGB = c(0.485, 0.456, 0.406)#Mean RGB image parameter
  std_RGB = c(0.229, 0.224, 0.225) #Standard RGB image parameter  
  #Image resize
  #print(dim(x))
  x<-torchvision::transform_to_tensor(x);
  x<-torchvision::transform_resize(x,c(64,64),interpolation=0);
  x<-torchvision::transform_normalize(x, mean=mean_RGB, std=std_RGB);
  return(x)
}

max_class <- function(tensor) {
  dimen = dim(tensor);
  tensor_index = torch_rand(dimen[1])
  for(i in 1:dimen[1]){
    t_row = tensor[i,];
    tensor_index[i]= t_row$argmax();
  }
  return(tensor_index)
}

#Para los resultados
real_label_class <- scan("data_source/real_label_class.txt", quiet = TRUE)
model_output_class<- scan("data_source/model_output_class.txt",quiet = TRUE)
matriz_confusion <- table(real_label_class, model_output_class)

TP <- rep(0, 7)
TN <- rep(0, 7)
FP <- rep(0, 7)
FN <- rep(0, 7)

for (i in 1:7) {
  TP[i] <- matriz_confusion[i, i]
  FN[i] <- sum(matriz_confusion[i, ]) - TP[i]
  FP[i] <- sum(matriz_confusion[, i]) - TP[i]
  TN[i] <- sum(diag(matriz_confusion)) - TP[i] - FP[i] - FN[i]
}

accuracy_class <- c()
precision_class <- c()
sensitivity_class <- c()
specificity_class <- c()
for (i in 1:7) {
  accuracy = (TP[i]+TN[i])/(TP[i]+TN[i]+FP[i]+FN[i])
  precision = TP[i]/(TP[i]+FP[i])
  sensitivity = TP[i]/(TP[i]+FN[i])
  specificity = TN[i]/(TN[i]+FP[i])
  
  accuracy_class = c(accuracy_class,accuracy)
  precision_class = c(precision_class,precision)
  sensitivity_class = c(sensitivity_class,sensitivity)
  specificity_class = c(specificity_class,specificity)
}


#-------------------------------------------

# ------------------------------------------
# UI
# ------------------------------------------

ui <- fluidPage(
  
  setBackgroundImage(src = "fondo1.png"),
  title = "BEDU: Análisis de datos con R",
  collapsible = TRUE,
  
  
  tabsetPanel(type = "tabs", 
              
    tabPanel(
        h4("Detalles del proyecto"), icon = icon("map", style="font-size: 30px"),
        div(
          style = "border-radius: 10px; background-color: rgba(218, 232, 245, 0.9); padding: 20px;",
            fluidRow(
              column(1),
              column(10,
                     br(),
                     h4("Módulo: Estadística y Programación con R", style ={"color:#FF5733;"}),
                     br(),br(),
                     h1("Detección de cancer de piel con base a imagenes.", style="color:#2874A6;font-weight:bold;"),
                     br(),
                     titlePanel("Problematica"),
                     h4("En México fallece una persona cada 12 horas por melanoma. La incidencia es alrededor de 2 mil 
                        casos por año. El melanoma solo representa el 1% de los casos de cancer de piel (a pesar de que causa la mayoria
                        de las muertes)"),
                     br(),
                     h4("Existe una gran variedad de lesiones cutaneas, no es facil identificar que lesiones dan signos
                        de malignidad y que lesiones son normales. Parte de los diagnosticos de lesiones 
                        malignas de la piel es herrado, siendo los falsos negativos los mas peligrosos, pues una lesión
                        puede considerarse segura cuando no lo es y evolucionar su malignidad."),
                     br(),
                     h4("Debido a lo mencionado surge la necesidad de una herramienta de facil acceso la cual pueda
                        proveer de una referencia solida para comenzar a dar seguimiento y diagnostico a una lesión cutanea
                        sospechosa."),
                     
                     titlePanel("Objetivos del proyecto"),
                     h4("Nuestro objetivo es entrenar una red neuronal de inteligencia artificial \"Deep learing\" 
                        que sea capaz de clasificar diversas lesiones cutáneas e indicar un porcentaje de la certeza que 
                        tiene el modelo sobre la clasificación proporcionada."),
                     br(),
                     h4("Nuestro modelo contempla 7 principales clases de lesiones, abreviadas de la siguiente forma:"),
                     tags$ul( 
                       tags$li(class="h4","Clase “nv”: Se refiere a los “nevos” comúnmente llamados lunares, son cúmulos benignos 
                           de células melanocíticas (las que dan color a la piel). La clase más común de lesión cutánea. "),
                       tags$li(class="h4","Clase “bkl”: Se refiere a la queratosis benigna, la acumulación benigna de la proteína 
                           queratina (la misma que compone nuestro cabello, pero en la piel se encarga de dar cierta dureza y protección)."),
                       tags$li(class="h4","Clase “vasc”: Se refiere a las lesiones vasculares no cancerígenas, son marcas rojizas 
                           producto de una lesión vascular (vasos sanguíneos superficiales que se inflaman)."),
                       tags$li(class="h4","Clase “df”: Se refiere a los dermatofibromas, pequeños tumores benignos que se producen por una producción excesiva de colágeno y otras proteínas."),
                       tags$li(style={"color:#B40E27"},class="h4","Clase “akiec”: Se refiere a lesiones cutánea con queratosis, es un tipo de carcinoma temprano no invasivo que puede tratarse fácilmente removiendo la lesión."),
                       tags$li(style={"color:#B40E27"},class="h4","Clase “bcc”: Se refiere a carcinoma basal, un tipo de cáncer de piel de crecimiento destructivo 
                           que raramente hace metástasis (propagación del cáncer por el cuerpo)."),
                       tags$li(style={"color:#B40E27"},class="h4","Clase “mel”: Se refiere al melanoma, una lesión cancerosa y agresiva que comúnmente hace metástasis.",),
                       br(),
                       br(),
                       br()
                     ),
                     fluidRow(
                       column(3,
                              img(src = "nv_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4("Ejemplo de lunar \"nv\"")
                       ),
                       column(3,
                              img(src = "bkl_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4("Ejemplo de queratosis \"bkl\"")
                       ),
                       column(3,
                              img(src = "vasc_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4("Ejemplo de vascularización \"vasc\"")
                       ),
                       column(3,
                              img(src = "df_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4("Ejemplo de dermatofibroma \"df\"")
                       ),
                       column(3,
                              img(src = "akiec_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4(style={"color:#B40E27"},"Ejemplo de lesión maligna de queratosis \"akiec\"")
                       ),
                       column(3,
                              img(src = "bcc_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4(style={"color:#B40E27"},"Ejemplo de carcinoma basal \"bcc\"")
                       ),
                       column(3,
                              img(src = "mel_example.jpg", width = "100%",style = "border: 4px solid black;"),
                              h4(style={"color:#B40E27"},"Ejemplo de melanoma \"mel\"")
                       )
                     )
              )
            )        
          )
      

             ),
    
    tabPanel(h4("Datos"), icon = icon("github", style="font-size: 30px"),
             fluidRow(

                      h2("Respecto a los datos",style = "text-align: center;"),
                      br()     
               
             ),
             
             fluidRow(
               column(1),
               column(10,
                      h4("Los datos fueron obtenidos del portal de Harvard Dataverse HAM10000."),
                      h4("La base de datos consta principalmente de 10015 imagenes, todas ellas clasificadas con base a 7 clases principales
                         de lesiones. Tambien se cuenta con un csv explicando las clases a las que pertenece cada imagen e información demografica
                         que constituyen la población de la base de datos
                         "),
                      br(), 
               )
             ),
             
             fluidRow(
               h2("Visualización de los datos",style = "text-align: center;"),
               br()     
               
             ),
             
             ###############tabla Dataset ###################
             wellPanel(
               titlePanel("Tabla de datos demograficos"),
               
               # Create a new Row in the UI for selectInputs
               fluidRow(
                 column(4,
                        selectInput("dx",
                                    "Diagnostico:",
                                    c("All",
                                      unique(as.character(metadata$dx))))
                 ),
                 column(4,
                        selectInput("dx_type",
                                    "Metodo de diagnostico:",
                                    c("All",
                                      unique(as.character(metadata$dx_type))))
                 ),
                 column(4,
                        selectInput("age",
                                    "Edad:",
                                    c("All",
                                      sort(unique(metadata$age))))
                        
                 ),
                 column(4,
                        selectInput("sex",
                                    "Sexo:",
                                    c("All",
                                      unique(as.character(metadata$sex))))
                 ),
                 column(4,
                        selectInput("localization",
                                    "Localización:",
                                    c("All",
                                      unique(as.character(metadata$localization))))
                 ),
               ),
               # Create a new row for the table.
               DT::dataTableOutput("metadata_table")
             ),

             ################################################
             
             ############### Histogramas ####################
             
             wellPanel(
               titlePanel("Proporción de los datos"),
               
               
               # Sidebar layout with input and output definitions ----
               sidebarLayout(
                 
                 # Sidebar panel for inputs ----
                 sidebarPanel(
                   
                   # Input: Select the random distribution type ----
                   radioButtons("dist", "Distribución de la variable:",
                                c("Diagnostico" = "dx",
                                  "Tipo de diagnostico" = "dx_type",
                                  "Edad" = "age",
                                  "Sexo" = "sex",
                                  "Localización" = "localization")),
                   
                   # br() element to introduce extra vertical spacing ----
                   br(),
                   
                 ),
                 
                 # Main panel for displaying outputs ----
                 mainPanel(
                   
                   # Output: Tabset w/ plot, summary, and table ----
                   tabsetPanel(type = "tabs",
                               tabPanel("Histograma", plotOutput("plot")),
                               tabPanel("Summary", verbatimTextOutput("summary"))
                   )
                   
                 )
               )               
               
               
             ),
             
             ################################################
             ),
    
    tabPanel(h4("Análisis"), icon = icon("flask", style="font-size: 30px"),
             ####### Explicación del preprocesamiento de datos #####
             h2("Analisis preambulatorio"),
             wellPanel(
               h2("Analisis inicial de la proporción de los datos"),
               sidebarLayout(
                 sidebarPanel(

                   radioButtons("data_h_ana", "Datos",
                                c("Diagnostico" = "dx",
                                  "Tipo de diagnostico" = "dx_type",
                                  "Edad" = "age",
                                  "Sexo" = "sex",
                                  "Localización" = "localization")),
                   
                   br(),
                   
                 ),
                 mainPanel(
                   
                   tabsetPanel(type = "tabs",
                               tabPanel("Proporción", plotOutput("plot_ana")),
                               tabPanel("Análisis", uiOutput("summary_ana"))
                   )
                   
                 )
               )
             ),
             wellPanel(
               h2("Visualizar valores atipicos en los atributos numericos"),
               fluidRow(
                 column(5,
                        h3("Parece que no hay valores atipicos
                           que puedan entorpecer el analisis."),
                        h3("Los datos se concentran en un rango
                           de 40 a 65 años.")
                        ),
                 column(7,
                        plotOutput("age_outlier")
                        )
                 
               )
             ),
             wellPanel(
               h2("Analisis de la relación del tipo de lesión con la edad"),
               sidebarLayout(
                 sidebarPanel(

                   radioButtons("data_dx_age", "Lesiones",
                                c("Queratosis actínica maligna - akiec" = "akiec",
                                  "Carcinoma basal - bcc" = "bcc",
                                  "Queratosis benigna - bkl" = "bkl",
                                  "Dermatofibroma - df" = "df",
                                  "Melanoma - mel" = "mel",
                                  "Nevo - nv" = "nv",
                                  "Lesión vascular - vasc" = "vasc")),

                   br(),
                   
                 ),
                 mainPanel(

                   tabsetPanel(type = "tabs",
                               tabPanel("Proporción", plotOutput("plot_data_dx_age")),
                               tabPanel("Análisis", uiOutput("ana_data_dx_age")),
                               tabPanel("Resumen de datos", verbatimTextOutput("summary_data_dx_age"))
                   )
                   
                 )
               )
             ),
             wellPanel(
               h2("Analisis de la distribución entre el tipo de lesión y su localización"),
               sidebarLayout(
                 sidebarPanel(
                   
                   radioButtons("data_dx_loc", "Lesiones",
                                c("Queratosis actínica maligna - akiec" = "akiec",
                                  "Carcinoma basal - bcc" = "bcc",
                                  "Queratosis benigna - bkl" = "bkl",
                                  "Dermatofibroma - df" = "df",
                                  "Melanoma - mel" = "mel",
                                  "Nevo - nv" = "nv",
                                  "Lesión vascular - vasc" = "vasc")),
                   
                   br(),
                   
                 ),
                 mainPanel(
                   
                   tabsetPanel(type = "tabs",
                               tabPanel("Proporción", plotOutput("plot_data_dx_loc")),
                               tabPanel("Análisis", uiOutput("ana_data_dx_loc")),
                               tabPanel("Resumen de datos", verbatimTextOutput("summary_data_dx_loc"))
                   )
                   
                 )
               )
             ),
             wellPanel(
               h2("Analisis de dispersión de los datos"),
               fluidRow(
                 column(5,
                        h3("La dispersión de los datos nos es util
                           para identificar posibles relaciones
                           y variables explcativas lineales o 
                           polinomiales."),
                        h3("Los datos no parecen sustentar un analisis
                           de regresión o clusterizado.")
                 ),
                 column(7,
                        plotOutput("disper")
                 )
                 
               )
             ),
            ################################################
            
            ),
    
    tabPanel(h4("Modelo"), icon = icon("cog", style="font-size: 30px"),
             wellPanel(
               titlePanel("Diagnosticar una imagen con el modelo"),
               
               fluidPage(
                 
                 # Copy the line below to make a file upload manager
                 fileInput("image_file", label = h3("Subir imagen"), accept = c('image/png', 'image/jpeg')),
                 
                 hr(),
                 fluidRow(column(4, verbatimTextOutput("image_upload"))),
                 br(),
               ),
                column(1),
               actionButton("process", "Procesar"),
                fluidRow(
                  column(4,
                         uiOutput("resultado_texto"),
                  ),
                  column(8,
                         plotOutput("uploaded_image")
                         )
                  
                ),
               fluidRow(
                 uiOutput("imagenes_repre")
                 ###########ASKDNASKDHNASDKJSANDOAIS
               )
             ),
             h2("¿Que es un modelo de red neuronal convolucional?"),
             wellPanel(
               fluidRow(
                 column(7,
                        h2("Modelos de redes neuronales"),
                        h3("Una red neuronal es un modelo computacional inspirado en la estructura 
                        del cerebro humano. Está compuesta por capas de nodos interconectados, 
                        llamados neuronas, que procesan información de entrada para realizar tareas 
                        de aprendizaje automático (¡machine learning! :D)."),
                        ),
                 column(5,
                        br(),
                        br(),
                        img(src = "deep_nn_example.png", width = "90%",style = "border: 4px solid black;"),
                        )
               ),
               fluidRow(
                 column(7,
                        h2("Modelos de redes neuronales convolucionales"),
                        h3("Las redes neuronales convolucionales 2D (CNN, por sus siglas en inglés) son un 
                        tipo especializado de red neuronal diseñada principalmente para procesar imágenes.
                        Están compuestas por capas de neuronas llamadas \"convolucionales\" que son capaces de 
                        aprender patrones y características específicas de las imágenes.
                        Las capas convolucionales en una CNN utilizan filtros (kernels) que se deslizan 
                        sobre la imagen original. Cada filtro extrae características como bordes, texturas o 
                        formas. Estas características se combinan en capas posteriores para formar representaciones 
                        más complejas de la imagen. Luego, se utilizan capas para reducir la dimensionalidad 
                        y resaltar las características más importantes."),
                 ),
                 column(5,
                        br(),
                        br(),
                        br(),
                        img(src = "convolution_example.png", width = "100%",style = "border: 4px solid black;"),
                 )
               ),

               
             ),
             h2("¿Por que un modelo neuronal convolucional y no otros?"),
             wellPanel(
               fluidRow(
                 column(7,
                        h2("Modelos de redes neuronales convolucionales VS otros"),
                        h3("Al ponderar que modelo emplear, se priorizaron 2 aspectos:"),
                        tags$ul( 
                          tags$li(class="h3","Costo computacional"),
                          tags$li(class="h3","Abstracción de caracteristicas generales en datos ruidosos"),
                        ),
                        h3("Los Random Forest y SVM son los otros modelos considerados para abordar este problema de
                           clasificación de datos 2D (imagenes), sin embargo, le modelo Random Forest se vuelven pesados y lentos
                           de entrenar con datos complejos, mas importante, son sensibles a datos ruidosos
                           (nuestros datos se caracterizan por tener caracteristicas 
                           ruidas y muy variadas)."),
                        h3("Respecto a las SVM, el calculo del hiperplano necesario para poder separar las caracteristicas
                           que definan los criterios para clasificar las imagenes seria computacionalmente costoso (nuestras imagenes
                           tienen dimensiones tres dimensiones, 2 espaciales y 1 en el espectro de color)."),
                 ),
                 column(5,
                        br(),
                        br(),
                        img(src = "model_compar.png", width = "90%",style = "border: 4px solid black;"),
                 )
               ),
             ),
             h2("¿Como fue este modelo entrenado?"),
             wellPanel(
               fluidRow(
                 column(6,
                        
                        h2("Tecnicas de Data Agumentation"),
                        br(),
                        h3("La técnica de data augmentation en imágenes para machine learning consiste
                      en crear nuevos datos a partir de un conjunto de datos existente. Esto se puede 
                      hacer aplicando una serie de transformaciones a las imágenes:"),
                        tags$ul( 
                          tags$li(class="h3","Rotar"),
                          tags$li(class="h3","Cambiar la perspectiva"),
                          tags$li(class="h3","Ajustar brillo"),
                          tags$li(class="h3","Ajusta contraste"),
                          tags$li(class="h3","Ajustes de color"),
                        ),
                        h3("La idea es aplicar transformaciones a la imagen que preserven las caracteristicas
                      de interes, para nuestro caso no realizamos ni cambio de perspectiva ni ajuste de color debido
                      a que la perspectiva puede alterar caracteristicas de forma (importantes para la clasificación)
                      y el color puede alterar pgimentaciones (importantes tambien), con ello ampliamos nuestros datos
                      y nos evitamos generar confusión en las caracteristicas."), 
                        br(),
                        h3("Estas tecnicas fueron aplicadas debido al fuerte desbalance de las clases de interes (diagnostico)."),
                        h3("El modelo fue entrenado con un total de 10500 imagenes (1500 imagenes cada clase).")
                        
                 ),
                 column(6,
                        br(),
                        br(),
                        br(),
                        br(),
                        br(),
                        img(src = "data_augmentation_example.png", width = "100%",style = "border: 4px solid black;"),  
                        
                        
                 )
               )
             ),
             
             
             
    ),
    tabPanel(h4("Resultados"), icon = icon("medal", style="font-size: 30px"),
             wellPanel(
               h2("Resultados del modelo para cada clase y general"),
               sidebarLayout(
                 sidebarPanel(
                   
                   radioButtons("results", "Resultados por lesión y general",
                                c("General" = "general",
                                  "Queratosis actínica maligna - akiec" = "akiec",
                                  "Carcinoma basal - bcc" = "bcc",
                                  "Queratosis benigna - bkl" = "bkl",
                                  "Dermatofibroma - df" = "df",
                                  "Melanoma - mel" = "mel",
                                  "Nevo - nv" = "nv",
                                  "Lesión vascular - vasc" = "vasc")),
                   
                   br(),
                   
                 ),
                 mainPanel(
                   
                   tabsetPanel(type = "tabs",
                               tabPanel("Resultados", plotOutput("plot_results_data")),
                               tabPanel("Análisis", uiOutput("ana_results_data"))
                   )
                   
                 )
               )
             ),
             wellPanel(
               h2("Matriz de confusión"),
               fluidRow(
                 column(5,
                        h3("Se ha observado un buen desempeño general, lo que se refleja en 
                        la precisión en la mayoría de las clases representadas. Sin embargo, 
                        se identifica una oportunidad para mejorar el modelo mediante la adquisición 
                        de más datos representativos."),
                        h3("El modelo proporciona una clasificación precisa en varias de las clases, 
                           con un alto número de predicciones correctas. No obstante, al profundizar en el análisis porcentual, 
                           se observa que algunas clases específicas muestran un desequilibrio en la precisión. 
                           Es probable que esto se deba a la falta de datos suficientes para estas clases menos representadas, 
                           lo que ha limitado la capacidad del modelo para aprender y generalizar patrones distintivos en esas áreas.")
                 ),
                 column(7,
                        plotOutput("matriz_confusion")
                 )
                 
               )
             ),

             ),
    
    tabPanel(h4("Equipo"), icon = icon("people-group", style="font-size: 30px"),
             tags$head(
               tags$style(HTML("
                      .team-member {
                        text-align: center;
                        margin-bottom: 20px;
                      }
                      .team-member img {
                        width: 200px;
                        height: auto;
                        border-radius: 50%;
                        margin-bottom: 10px;
                      }
                      .links a {
                        display: inline-block;
                        margin-right: 10px;
                      }
                    "))
             ),
             br(),
             br(),
             fluidRow(
               column(6,
                      div(class = "team-member",
                          tags$img(src = "cesar_ochoa.jpg", alt = "César Alberto Ochoa Ávila"),
                          h3("César Alberto Ochoa Ávila - Estudiante de Ingenieria Biomédica"),
                          div(class = "links",
                              a(href = "https://github.com/Cesar-8A", target = "_self", "GitHub"),
                              a(href = "https://www.linkedin.com/in/cesar-alberto-ochoa-avila-32183a24a/", target = "_self", "LinkedIn"),
                              a(href = "mailto:cesar.ochoa7682@alumnos.udg.mx", "Correo")
                          )
                      )
               ),
               column(6,
                      div(class = "team-member",
                          tags$img(src = "gamaliel_naranjo.jpg", alt = "Gamaliel Osvaldo Naranjo Bernal"),
                          h3("Gamaliel Osvaldo Naranjo Bernal - Ingeniero Civil"),
                          div(class = "links",
                              a(href = "https://github.com/GamalielNB", target = "_self", "GitHub"),
                              a(href = "https://www.linkedin.com/in/gamalielnaranjo", target = "_self", "LinkedIn"),
                              a(href = "mailto:gamalielnaranjo@gmail.com", "Correo")
                          )
                      )
               ),

             )
             )
  
    )
)







  # ------------------------------------------
  # Server
  # ------------------------------------------
  server <- function(input, output, session) {
    #output Tabla
    output$dfVisitantes <- renderDT({
      if(input$pais == "Todos"){
        japan_df <- df
      }else {
        japan_df <- df[df$Pais == input$pais,]
      }
      dfFiltrado <- japan_df
    })
    
    ############ Dataset Table/Datos ####################
    
    # Filter data based on selections
    output$metadata_table <- DT::renderDataTable(DT::datatable({
      data <- metadata
      if (input$dx != "All") {
        data <- data[data$dx == input$dx,]
      }
      if (input$dx_type != "All") {
        data <- data[data$dx_type == input$dx_type,]
      }
      if (input$age != "All") {
        data <- data[data$age == input$age,]
      }
      if (input$sex != "All") {
        data <- data[data$sex == input$sex,]
      }
      if (input$localization != "All") {
        data <- data[data$localization == input$localization,]
      }
      data
    }))
    ###############################################
    
    ############ Histogramas/Datos ######################
    
    output$plot <- renderPlot({
      dist <- input$dist
      
      if(dist == "dx"){
        freq <- table(metadata$dx)
        barplot(freq, main = "Histograma de diagnosticos", xlab = "Diagnosticos", ylab = "Frecuencia")
        
      }
      if(dist == "dx_type"){
        freq <- table(metadata$dx_type)
        barplot(freq, main = "Histograma de tipos de diagnosticos", xlab = "Tipos de diagnosticos", ylab = "Frecuencia")
      }
      if(dist == "age"){
        freq <- table(metadata$age)
        barplot(freq, main = "Histograma de edades", xlab = "Edades", ylab = "Frecuencia")
      }
      if(dist == "sex"){
        freq <- table(metadata$sex)
        barplot(freq, main = "Histograma de sexos", xlab = "Sexos", ylab = "Frecuencia")
      }
      if(dist == "localization"){
        freq <- table(metadata$localization)
        barplot(freq, main = "Histograma de localización", xlab = "Localizaciones", ylab = "Frecuencia")
      }
    })
    
    # Generate a summary of the data ----
    output$summary <- renderPrint({
      dist <- input$dist
      if(dist == "dx"){
        x <- summary(metadata$dx)
      }
      if(dist == "dx_type"){
        x <- summary(metadata$dx_type)
      }
      if(dist == "age"){
        x <- summary(metadata$age)
      }
      if(dist == "sex"){
        x <- summary(metadata$sex)
      }
      if(dist == "localization"){
        x <- summary(metadata$localization)
      }
      x
    })
  
    
    
    ###############################################
    
    ############## Histogramas /Analisis ###################
    #--------------- Overview
    output$plot_ana <- renderPlot({
      dist <- input$data_h_ana
      if(dist == "dx"){
        percentage_data <- metadata %>%
          count(dx) %>%
          mutate(percentage = prop.table(n) * 100)
        
        p <- ggplot(percentage_data, aes(x = dx, y = percentage)) +
          geom_bar(stat = "identity", fill = "#00008B") +
          geom_text(aes(label = paste0(round(percentage, 1), "%")), 
                    vjust = -0.5, size = 5, color = "black") +
          labs(title = "Diagnostico", 
               x = "Tipo de lesión", y = "Porcentaje") +
          theme(axis.text.x = element_text(size= 15,angle = 0, vjust = 0.5),
                axis.title = element_text(size = 15, face = "bold"),
                plot.title = element_text(size = 30, face = "bold"))
        return(p)
      }
      if(dist == "dx_type"){
        percentage_data <- metadata %>%
          count(dx_type) %>%
          mutate(percentage = prop.table(n) * 100)
        
        p <- ggplot(percentage_data, aes(x = dx_type, y = percentage)) +
          geom_bar(stat = "identity", fill = "#00008B") +
          geom_text(aes(label = paste0(round(percentage, 1), "%")), 
                    vjust = -0.5, size = 5, color = "black") +
          labs(title = "Metodo de diagnostico", 
               x = "Metodo de diagnostico", y = "Porcentaje") +
          theme(axis.text.x = element_text(size= 15,angle = 0, vjust = 0.5),
                axis.title = element_text(size = 15, face = "bold"),
                plot.title = element_text(size = 30, face = "bold"))
        return(p)
      }
      if(dist == "age"){
        percentage_data <- metadata %>%
          count(age) %>%
          mutate(percentage = prop.table(n) * 100)
        
        p <- ggplot(percentage_data, aes(x = age, y = percentage)) +
          geom_bar(stat = "identity", fill = "#00008B") +
          geom_text(aes(label = paste0(round(percentage, 1), "%")), 
                    vjust = -0.5, size = 5, color = "black") +
          labs(title = "Edad", 
               x = "Edad", y = "Porcentaje") +
          theme(axis.text.x = element_text(size= 15,angle = 0, vjust = 0.5),
                axis.title = element_text(size = 15, face = "bold"),
                plot.title = element_text(size = 30, face = "bold"))
        return(p)
      }
      if(dist == "sex"){
        percentage_data <- metadata %>%
          count(sex) %>%
          mutate(percentage = prop.table(n) * 100)
        
        p <- ggplot(percentage_data, aes(x = sex, y = percentage)) +
          geom_bar(stat = "identity", fill = "#00008B") +
          geom_text(aes(label = paste0(round(percentage, 1), "%")), 
                    vjust = -0.5, size = 5, color = "black") +
          labs(title = "Sexo", 
               x = "Sexo", y = "Porcentaje") +
          theme(axis.text.x = element_text(size= 15,angle = 0, vjust = 0.5),
                axis.title = element_text(size = 15, face = "bold"),
                plot.title = element_text(size = 30, face = "bold"))
        return(p)
      }
      if(dist == "localization"){
        percentage_data <- metadata %>%
          count(localization) %>%
          mutate(percentage = prop.table(n) * 100)
        
        p <- ggplot(percentage_data, aes(x = localization, y = percentage)) +
          geom_bar(stat = "identity", fill = "#00008B") +
          geom_text(aes(label = paste0(round(percentage, 1), "%")), 
                    vjust = -0.5, size = 5, color = "black") +
          labs(title = "Localización de la lesión", 
               x = "Localización", y = "Porcentaje") +
          theme(axis.text.x = element_text(size= 10,angle = 10, vjust = 0.5),
                axis.title = element_text(size = 15, face = "bold"),
                plot.title = element_text(size = 30, face = "bold"))
        return(p)
      }
      
      
    })
    
    # Generate a summary of the data ----
    output$summary_ana <- renderUI({
      dist <- input$data_h_ana
      if(dist == "dx"){
         x<-div(
            h3("Se cuenta con una proporción relativamente desbalanceada en cuanto a los tipos de lesiones
                presentes en la base de datos, esto se debe principalmente a la rareza misma de las lesiones de piel, pues
                dentro de ese paradigma, los lunares, queratosis y otras lesiones benignas son mucho mas comunes que
                las lesiones malignas como melanoma o carcinoma.
                "),
         )
         return(x)
      }
      if(dist == "dx_type"){
        x<-div(
          h3("Los metodos de diagnostico mas confiables son los histologicos, es posible que por eso sea el mas comun en 
                este dataset, sin embargo, todos los datos estan
                confirmados por una serie de expertos.
                "),
        )
        return(x)
      }
      if(dist == "age"){
        x<-div(
          h3("Una distribución normal en la edad, con un promedio de edad de 51 años, y una mediana de 50, el datase esta formado 
                principalmente por gente adulta madura.
                "),
        )
        return(x)
      }
      if(dist == "sex"){
        x<-div(
          h3("La proporción del sexo femenino es ligeramente menor a la del masculino, apenas un 8.6%, esto no representa un desbalance 
              que impacte en los futuros analisis, debido al volumen total de datos con el que se cuenta (10015 datos, sin descartar
              valores NA en edad e identificadores de sexo desconocido).
                "),
        )
        return(x)
      }
      if(dist == "localization"){
        x<-div(
          h3("Las localizaciones corporales mas comunes para las lesiones identificadas son la espalda y extremidades inferiores, 
          juntas representan un 42.9% del total de las lesiones.
                "),
        )
        return(x)
      }
      
    })
    
    #----------- Check Age outlier
    output$age_outlier <- renderPlot({
      p <- ggplot(metadata, aes(y = age)) +
        geom_boxplot(color="red") +
        labs(title = "Rango de edad", y = "Rango") +
        theme(axis.text.x = element_text(size= 10,angle = 10, vjust = 0.5),
              axis.title = element_text(size = 15, face = "bold"),
              plot.title = element_text(size = 30, face = "bold")) 
      return(p)
    })
    
    
    #----------- Dx_Age
    output$plot_data_dx_age <- renderPlot({
      dist <- input$data_dx_age
      classes <- sort(unique(metadata$dx))
      for (class in classes){
        if(dist == class){
          df_filtrado <- metadata[metadata$dx == class, c("age", "sex")]
          
          # Crate a list of aesthetics
          p <- ggplot(df_filtrado, aes(x = age, fill = sex)) +
            geom_bar(position = "dodge", color = "black") +
            geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5,  hjust=0.3, size = 2.5) +
            labs(title = paste("Distribución por edad - ", class), x = "Age", y = "Count") +
            theme(axis.text.x = element_text(size= 20,angle = 0, vjust = 0.5),
                  axis.title = element_text(size = 15, face = "bold"),
                  plot.title = element_text(size = 30, face = "bold")) +
            scale_fill_brewer(palette = "Blues") +
            guides(fill = guide_legend(title = "Sex")) +
            coord_cartesian(clip = "off") 
          return(p)
        }
      }
    })
    
    output$ana_data_dx_age <- renderUI({
      dist <- input$data_dx_age
      if(dist == "akiec"){
        x<-div(
          h3("Se cuenta con un rango de edades maduro, de entre 30 y 85 años, con una media de 66 años. Hay 327 miembros con
          esta lesión y una clara incidencia mayoritaria en el sexo masculino."),
        )
        return(x)
      }
      if(dist == "bcc"){
        x<-div(
          h3("Se cuenta con un amplio rango de edades, de 20 a 85 años, con una media de 66 años. Es destacable que hay una incidencia
             relativamente similar en el sexo femenino y masculino a exceción del rango de 70 a 80 años, donde la incidencia en el sexo
             masculino es de casi el doble que en el femenino."),
        )
        return(x)
      }
      if(dist == "bkl"){
        x<-div(
          h3("Se cuenta con un amplio rango, de entre 0 (recien nacidos) a 85 años, es notable la existencia de valores atipicos,
             la tendencia generla indica una mayor indicendia en edades avanzadas pues es una lesión comunmente producida por daño
             de radiación solar y fricción fisica constante."),
        )
        return(x)
      }
      if(dist == "df"){
        x<-div(
          h3("Se cuenta con una distribución relativamente normal, con un rango de entre 25 a 80 años, es destacable la alta incidencia
             de esta lesión en el sexo femenino en el rango de edad de 30 años, es posible que se pueda atribuir a un proceso biologico hormonal."),
          br(),
          h3("Los analisis medicos quedan fuera del alcance de este proyecto.")
        )
        return(x)
      }
      if(dist == "mel"){
        x<-div(
          h3("Se cuenta con una distribución normal en ambos sexos, con un rango de edades de entre 5 a 85 años, con una mayor incidencia en el
             sexo masculino, aparentemente el progreso de la edad contribuye a la incidencia hasta los 70 años, donde la incidencia disminuye.
             Es destacable la presencia de un valor atipico en un recien nacido."),
        )
        return(x)
      }
      if(dist == "nv"){
        x<-div(
          h3("El tipo de lesión mas comun, cuenta con una distribución normal en ambos sexos y esta presente en todas las edades.
             La incidencia es mayor en el sexo femenino en el rango de edades de 0 a 50 años (siendo 46 años la media) y la incidencia
             es mayor de 50 a 85 años en el sexo masculino."),
        )
        return(x)
      }
      if(dist == "vasc"){
        x<-div(
          h3("Se cuenta con una distribución irregular que no parece indicar un patron definido, es una lesión presente en todas las edades.
             Es destacable el predominio en el sexo masculino en el rango de 5 a 20 años."),
        )
        return(x)
      }
    })
    
    output$summary_data_dx_age <- renderPrint({
      dist <- input$data_dx_age
      classes <- sort(unique(metadata$dx))
      for (class in classes){
        if(dist == class){
          df_filtrado <- metadata[metadata$dx == class, c("age", "sex")]
          p <- summary(df_filtrado)
          return(p)
        }
      }
    })
    
    #----------- Dx_Loc
    output$plot_data_dx_loc <- renderPlot({
      dist <- input$data_dx_loc
      classes <- c("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
      
      for (class in classes) {
        # Filter data for keeping age and sex distributions
        if(dist==class){
          df_filtrado <- metadata[metadata$dx == class, c("localization","age")]
          
          p<- ggplot(df_filtrado, aes(x = localization)) +
            geom_bar(position = "dodge", color = "black") +
            geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5,  hjust=0.3, size = 4) +
            labs(title = paste("Distribución de la lesión -", class), x = "Localización") +
            theme(axis.text.x = element_text(size= 20,angle = 20, vjust = 0.5),
                  axis.title = element_text(size = 15, face = "bold"),
                  plot.title = element_text(size = 30, face = "bold")) +
            scale_fill_brewer(palette = "Blues") +
            guides(fill = guide_legend(title = "Sex")) +
            coord_cartesian(clip = "off")           
          return(p)
        }
      }
      
    })

    output$ana_data_dx_loc <- renderUI({
      dist <- input$data_dx_loc
      if(dist == "akiec"){
        x<-div(
          h3("Se cuenta con una distribución irregular, con una mayor incidencia en el rostro."),
        )
        return(x)
      }
      if(dist == "bcc"){
        x<-div(
          h3("Se cuenta con una distribución irregular, la proporción de datos
             indica una mayor presencia en la cara y extremidades (superiores e inferiores)."),
        )
        return(x)
      }
      if(dist == "bkl"){
        x<-div(
          h3("Hay una clara presencia mayoritaria en la espalda y cara, y una baja presencia
             en pies y manos."),
        )
        return(x)
      }
      if(dist == "df"){
        x<-div(
          h3("Hay una mayor incidencia en las extremidades inferiores y superiores
             habiendo casi nada de presencia en el resto de localizaciones.
             Debido a que la muestra de dermatofibroma es pequeño, no sustenta mayores
             conclusiones."),
        )
        return(x)
      }
      if(dist == "mel"){
        x<-div(
          h3("Existe una alta incidencia en espalda y extremidades superiores e inferiores."),
        )
        return(x)
      }
      if(dist == "nv"){
        x<-div(
          h3("Es la lesión mas presente en el dataset, indica una presencia mayoritara en extremidades,
             espalda, abdomen y torzo. A pesar de que el resto de localizaciones no tienen tanta incidencia
             aun asi cuentan con una cantidad considerable."),
        )
        return(x)
      }
      if(dist == "vasc"){
        x<-div(
          h3("Se cuentan con pocas muestras de esta lesión. Tiene mayor incidencia en el torzo en general."),
        )
        return(x)
      }
    })
    
    output$summary_data_dx_loc <- renderPrint({
      dist <- input$data_dx_loc
      classes <- c("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
      for (class in classes){
        if(dist == class){
          df_filtrado <- metadata[metadata$dx == class, c("localization", "dx","sex")]
          p <- summary(df_filtrado)
          return(p)
        }
      }
    })  
    
    #----------- Disper
    output$disper <- renderPlot({
      metadata_encoded <- metadata
      metadata_encoded$localization <- as.numeric(factor(metadata$localization))
      
      p <- ggplot(metadata_encoded, aes(x = age, y = dx, color=localization)) +
        geom_point(size = 4) +
        scale_color_gradient(low = "blue", high = "red") +
        labs(title = "Gráfico de dispersión", x = "Edad", y = "Diagnostico", color = "Localización")
      return(p)
    })
    
    #######################################################
    
    
    ##################### Procesar imagen ########################
    
    pasar_modelo <- function() {
      available_classes = c("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc")
      if(is.null(input$image_file)){
        return(NULL)
      }
      ruta <- input$image_file$datapath
      ruta = gsub("\\\\", "/", ruta)
      imagen = image_read(ruta)
      imagen = compose_substitute(imagen)
      #Crear un bache para ingresar la imagen
      molde = torch_rand(1,3,64,64)
      molde[1,,,] = imagen
      #Ingresar la imagen
      softmax_layer = nn_softmax(dim=2)
      result = softmax_layer(modelo_cargado(molde))
      #Obtener clase perteneciente y guardar probabilidades
      probabilidad = as_array(result)
      result = max_class(result);
      result = available_classes[as_array(result)];
      final_result = c(result,probabilidad)

      return(final_result)
    }
    
    observeEvent(input$process, {
      # Llamar a la función de cálculo
      resultado <- pasar_modelo()
      if(is.null(resultado)){
        p(style={"color:#B40E27"},paste("Carga una imagen de formato jpg o png para procesar"))
      }else{
        output$resultado_texto <- renderUI({
          
          available_classes = sort(unique(metadata$dx))
          div(
            if(sum(resultado[1] == c("mel","bcc","akiec"))){
              div(
                h4(style={"color:#B40E27"},paste("El modelo cree que tu lesion es de tipo: ", resultado[1])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[2])*100,digits=2)),'%'," de certeza en ", available_classes[1])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[3])*100,digits=2)),'%'," de certeza en ", available_classes[2])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[4])*100,digits=2)),'%'," de certeza en ", available_classes[3])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[5])*100,digits=2)),'%'," de certeza en ", available_classes[4])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[6])*100,digits=2)),'%'," de certeza en ", available_classes[5])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[7])*100,digits=2)),'%'," de certeza en ", available_classes[6])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[8])*100,digits=2)),'%'," de certeza en ", available_classes[7]))
              )
              
            }
            else{
              div(
                h4(paste("El modelo cree que tu lesion es de tipo: ", resultado[1])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[2])*100,digits=2)),'%'," de certeza en ", available_classes[1])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[3])*100,digits=2)),'%'," de certeza en ", available_classes[2])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[4])*100,digits=2)),'%'," de certeza en ", available_classes[3])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[5])*100,digits=2)),'%'," de certeza en ", available_classes[4])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[6])*100,digits=2)),'%'," de certeza en ", available_classes[5])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[7])*100,digits=2)),'%'," de certeza en ", available_classes[6])),
                h4(paste("El modelo tiene un ",(round(as.numeric(resultado[8])*100,digits=2)),'%'," de certeza en ", available_classes[7]))
              )
            },
          )
          
        })
        output$uploaded_image <- renderPlot({
          df <- data.frame(
            type = c("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"),  # Etiquetas para cada valor en el vector
            value = (round(as.numeric(resultado[2:8])*100,digits=2))
          )
          df <- df %>% 
            mutate(porcentaje = value / sum(value) * 100)
          
          p <- ggplot(df, aes(x = "", y = value, fill = type)) +
            geom_bar(stat = "identity", width = 1) +
            coord_polar("y", start = 0) +
            theme_void() +
            scale_fill_brewer(palette = "Set2") +
            geom_text(aes(label = paste0(round(porcentaje), "%")), position = position_stack(vjust = 0.5))+
            theme(legend.position = "right", legend.text = element_text(size = 20)) 
            
          return(p)
        })
        
        output$imagenes_repre <- renderUI({
          div(
            h3(paste0("¿Por que el modelo piensa que tengo ",resultado[1],"?")),
            h4("El objetivo del modelo es abstraer los criterios ABCD (usado para la evaluación
               de lesiones potencialmente malignas)."),
            tags$ul( 
              tags$li(class="h4","A de Asimetría: Lesiones asimetricas (de forma irregular)."),
              tags$li(class="h4","B de Bordes irregulares: Lesuines con birdes no definidos."),
              tags$li(class="h4","C de Color: Lesuines multicolor o de un color extravagante."),
              tags$li(class="h4","D de Diámetro: Lesiones grandes."),
            ),
            h4("Lesiones que cumplan con una o varias de las caracteristicas mencionadas deben
               ser analizadas por un experto, pues son potenciales a malignidad."),
            br(),
            h4("El modelo no debe ser tomado como una referencia medica, si no con fines informativos.
               Los datos con los que se entrenó el modelo fueron datos representativos pero ruidosos, por
               lo que los resultados del modelo no son completamente fiables."),
            h3("Observa las siguientes imagenes representativas con las cuales el modelo fue entrenado:"),
            tags$div(
              id = "gallery",
              style = "display: flex; justify-content: space-between;",
              tags$div(
                id = "akiec",
                tags$img(src = "akiec_example.jpg", alt = "akiec",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Queratosis Actinica"),
                  tags$p("Akiec")
                )
              ),
              tags$div(
                id = "bcc",
                tags$img(src = "bcc_example.jpg", alt = "bcc",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Carcinoma Basal"),
                  tags$p("bcc")
                )
              ),
              tags$div(
                id = "bkl",
                tags$img(src = "bkl_example.jpg", alt = "bkl",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Nevo (lunar)"),
                  tags$p("bkl")
                )
              ),
              tags$div(
                id = "df",
                tags$img(src = "df_example.jpg", alt = "bkl",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Dermatofibroma"),
                  tags$p("df")
                )
              ),
              tags$div(
                id = "mel",
                tags$img(src = "mel_example.jpg", alt = "bkl",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Melanoma"),
                  tags$p("mel")
                )
              ),
              tags$div(
                id = "nv",
                tags$img(src = "nv_example.jpg", alt = "bkl",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Nevo (lunar)"),
                  tags$p("nv")
                )
              ),
              tags$div(
                id = "vasc",
                tags$img(src = "vasc_example.jpg", alt = "bkl",width = "75%",style = "border: 4px solid black;"),
                tags$div(
                  id = "caption",
                  tags$h3("Lesión vascular"),
                  tags$p("vasc")
                )
              )
            ),
            h4("Si crees tener una lesión potencial a malignidad consulta a tu medico.")
          )
        })
      }
      #output$uploaded_image <- renderImage({
      #  ruta <- input$image_file$datapath
      #  ruta = gsub("\\\\", "/", ruta)
      #  imagen = image_read(ruta)
      #  #imagen = image_data(imagen)
      #  outfile = tempfile(fileext = ".png")
      #  image_write(imagen, path = outfile)
      #  
      #  list(src = outfile,
      #       contentType = "image/png",
      #       alt = "This is alternate text")
      #},deleteFile = TRUE)
    })
    
    output$image_upload <- renderPrint({
      
      if(!is.null(str(input$image_file))){
        str(input$image_file)
      }
    })
    ###############################################
    
    ############ Graficas de resultados  ###########
    
    output$plot_results_data <- renderPlot({
      dist <- input$results
      classes <- c("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc","general")
      i=1
      for (class in classes) {
        # Filter data for keeping age and sex distributions
        if(dist==class){
          if(dist=="general"){
            porcentajes <- c(mean(accuracy_class), mean(precision_class), mean(sensitivity_class), mean(specificity_class))
          }
          else{
            porcentajes <- c(accuracy_class[i], precision_class[i], sensitivity_class[i], specificity_class[i])
          }
          df <- data.frame(
            categoria = c("Accuracy", "Precision", "Sensitivity", "Specificity"),
            valor = round(porcentajes*100, 2)
          )
          
          p<- ggplot(df, aes(x = categoria, y = valor)) +
            geom_bar(stat = "identity", fill = "#042343", width = 0.5) +
            geom_text(aes(label = paste0(valor, "%")), vjust = -0.5, size = 4) +
            labs(x = paste0("Desempeño ", class), y = "Porcentaje") +
            scale_fill_brewer(palette = "Blues") +
            ylim(0, 100)+
            coord_cartesian(clip = "off")+
            theme(axis.text.x = element_text(size= 20,angle = 0, vjust = 0.5),
                  axis.title = element_text(size = 15, face = "bold"),
                  plot.title = element_text(size = 30, face = "bold")) 
          
          return(p)
        }
        i=i+1
      }
      
    })
    
    output$ana_results_data <- renderUI({
      dist <- input$results
      if(dist=="general"){
        x<-div(
          h3("El modelo cuenta con una buena exactitud, implicando que el modelo tiene un rendimiento bueno
             al clasificar imagenes."),
          h3("Cuenta con una precisión moderada lo que significa que identifica relativamente
             bien cuando cada imagen como la clase a la que pertenece realmente"),
          h3("Cuenta con baja sensibilidad, lo que significa que clasifica algunas imagenes a las clases incorrectas"),
          h3("Cuenta con baja especificidad, lo que significa que sabe cuando una imagen no pertenece a una lesión")
        )
        return(x)
      }
      if(dist == "akiec"){
        x<-div(
          h3("Relativamente buena exactitud. El modelo sabe diferenciar esta clase de lesión por lo general"),
          h3("Cuenta con una precisión moderada lo que significa que el modelo algunas veces identifica
             otro tipo de lesión como akiec."),
          h3("Cuenta con sensibilidad moderada, el modelo suele reconocer correctamente cuando se trata de esta lesión."),
          h3("Cuenta con alta especificidad, el modelo suele diferenciar correctamente imagenes que no corresponden a esta clase.")
        )
        return(x)
      }
      if(dist == "bcc"){
        x<-div(
          h3("Relativamente buena exactitud. El modelo sabe diferenciar esta clase de lesión por lo general"),
          h3("Cuenta con una precisión moderada lo que significa que el modelo algunas veces identifica
             otro tipo de lesión como bcc"),
          h3("Cuenta con sensibilidad moderada, el modelo suele reconocer correctamente cuando se trata de esta lesión."),
          h3("Cuenta con alta especificidad, el modelo suele diferenciar correctamente imagenes que no corresponden a esta clase.")
        )
        return(x)
      }
      if(dist == "bkl"){
        x<-div(
          h3("Cuenca con buena exactitud. El modelo sabe diferenciar esta clase de lesión por lo general"),
          h3("Cuenta con una buena precisión moderada lo que significa que el modelo no suele identificar otras lesiones como bkl"),
          h3("Cuenta con sensibilidad moderada, el modelo suele reconocer correctamente cuando se trata de esta lesión."),
          h3("Cuenta con alta especificidad, el modelo suele diferenciar correctamente imagenes que no corresponden a esta clase.")
        )
        return(x)
      }
      if(dist == "df"){
        x<-div(
          h3("Cuenca con exactitud moderada. El modelo sabe diferenciar esta clase de lesión por lo general"),
          h3("Cuenta con una buena precisión lo que significa que el modelo puede identificar otras lesiones como df"),
          h3("Cuenta con sensibilidad baja, el modelo puede tener problemas para reconocer correctamente cuando se trata de esta lesión."),
          h3("Cuenta con alta especificidad, el modelo suele diferenciar correctamente imagenes que no corresponden a esta clase.")
        )
        return(x)
      }
      if(dist == "mel"){
        x<-div(
          h3("Cuenca con exactitud alta, el modelo sabe diferenciar esta clase de lesión"),
          h3("Cuenta con una precisión moderada lo que significa que el modelo no suele identificar otras lesiones como mel"),
          h3("Cuenta con sensibilidad alta, el modelo puede reconocer correctamente cuando se trata de esta lesión."),
          h3("Cuenta con alta especificidad, el modelo puede diferenciar correctamente imagenes que no corresponden a esta clase.")
        )
        return(x)
      }
      if(dist == "nv"){
        x<-div(
          h3("Cuenca con exactitud alta, el modelo sabe diferenciar esta clase de lesión"),
          h3("Cuenta con una precisión moderada lo que significa que el modelo no suele identificar otras lesiones como nv"),
          h3("Cuenta con sensibilidad moderada, el modelo suele reconocer correctamente cuando se trata de esta lesión."),
          h3("Cuenta con alta especificidad, el modelo puede diferenciar correctamente imagenes que no corresponden a esta clase.")
        )
        return(x)
      }
      if(dist == "vasc"){
        x<-div(
          h3("¡Excelente desmpeño en general, el modelo sabe cuando se trata de un vasc casi siempre!"),
        )
        return(x)
      }
    })    
    
    output$matriz_confusion <- renderPlot({
      classes <- c("akiec", "bcc", "bkl", "df", "mel", "nv", "vasc","general")
      matriz_confusion_df <- as.data.frame(as.table(matriz_confusion))
      
      matriz_confusion_df$real_label_class <- classes[matriz_confusion_df$real_label_class]
      matriz_confusion_df$model_output_class <- classes[matriz_confusion_df$model_output_class]
      
      # Plot confusion matrix
      p <- ggplot(matriz_confusion_df, aes(x = real_label_class, y = model_output_class, fill = Freq)) +
        geom_tile() +
        geom_text(aes(label = Freq), vjust = 1) +
        scale_fill_gradient(low = "lightblue", high = "darkblue") +
        labs(title = "Confusion Matrix",
             x = "Real class",
             y = "Model output class")
      theme(axis.title = element_text(size = 25, face = "bold"),
            plot.title = element_text(size = 30, face = "bold")) 
      return(p)
    })

    ###############################################
  }
  



  shinyApp(ui, server)
  
  
  
  
  
  
  
  
  
  
  
  
  