#------------------- DASHBOARD STRUCTURE----------------------------------------

# Load packages ----------------------------------------------------------------
library(DT) # for data table
library(sf) # to read geoJSON files
library(readr) # to read csv files
library(dplyr) #for data manipulation
library(ggplot2) # for plots
library(leaflet) # for map
library(leaflet.extras) # for map
library(purrr) # for R to understand `%||%`
library(RColorBrewer) # for colors
library(magrittr) # for R to understand '%>%'
library(plotly) # for the plots and charts
library(shiny) # to run the shiny app
library(bslib) # for themes
library(shinythemes) # for themes
library(shinyWidgets) # slider, check boxes, etc

# 1. Load data -----------------------------------------------------------------

#Load the GeoJSON file containing country boundaries
outline <- st_read("C:\\Users\\mdroogleverfortuyn\\OneDrive - Rode Kruis\\Documenten\\GitHub\\RIPOSTE\\TdB Incidence Choléra\\Data\\geoBoundaries-CMR-ADM1.geojson")

#Load the .csv file containing the attribute information
incidence <- read.csv("C:\\Users\\mdroogleverfortuyn\\OneDrive - Rode Kruis\\Documenten\\GitHub\\RIPOSTE\\TdB Incidence Choléra\\Data\\Slider_Regional_Incidence.csv")

#For the time slider, it is important to make sure the date is in the right format
incidence$start_date <- as.Date(incidence$start_date, format="%d-%m-%Y")

#Obtaining the unique values of dates in the date range
date_range <- unique(incidence$start_date)

#Defining deaths per region
deaths <- count(incidence, ADM1_FR, wt = deaths)

#Listing the different regions in Cameroon
regions <- unique(incidence$ADM1_FR)

# Assuming start_date is a character vector with "DD-MM-YYYY" format
incidence <- mutate(incidence,
                    start_date = as.Date(start_date, format = "%d-%m-%Y"))

#Merging the .csv file and GeoJSON in the leaflet map, we need to merge data
merged <- merge(outline, incidence, by="ADM1_FR")

# 2. Define UI - User Interface ------------------------------------------------

ui <- 
  
  #Navbar talks about the various panels that would be visible in the application
  navbarPage(h4("Géovisualisation du choléra au Cameroun"), collapsible = T, 
             
             #---------------- Cholera tab ---------- UI code --------------- 
             
             tabPanel(h5("Carte de l'incidence du choléra"),
                      div(class="outer",
                          #Linking the CSS file for the styling of the dashboard
                          tags$head(includeCSS("styles.css")),
                          
                          leafletOutput("mymap", width="100%", height="100%"),
                          
                          absolutePanel(h3("Carte de l'incidence du choléra"), 
                                        p("La carte visualise l'incidence des cas de choléra au Cameroun au cours de la période représentée par le curseur."), 
                                        p("Le curseur peut être joué pour visualiser la tendance ou être utilisé pour sélectionner la date pour laquelle l'utilisateur a besoin d'informations :"),
                                        
                                        id = "controls", class = "panel panel-default",
                                        top = 100, left = 75, width = 400, fixed = T,
                                        draggable = F, height = "auto",
                                        
                                        sliderTextInput(inputId = "date_slider",
                                                        label = "Sélectionner la date :",
                                                        choices = format(unique(incidence$start_date), "%d %b %y"),
                                                        selected = format((min(incidence$start_date)), "%d %b %y"),
                                                        grid = F,
                                                        animate = animationOptions(interval = 3000, loop = FALSE)
                                        ),
                                        
                                        #adding the logos to the dashboard
                                        absolutePanel(class = "card",
                                                      bottom = 20, left = 60, fixed=TRUE, draggable = FALSE, height = "auto",
                                                      img(src='logo.png', height = 80, width = 320))
                                        
                          ))),
             
             #------------ Outbreak Plots tab ---------- UI code -------------------------
             
             tabPanel(h5("Graphiques de l'épidémie"), 
                      
                      tabsetPanel(
                        
                        tabPanel("Graphiques en barres et lignes",
                                 
                                 h4("Nombre total de cas et de décès dus au choléra au Cameroun par région"),
                                 p("Dans cet onglet, deux graphiques ont été créés pour comprendre l'épidémie de choléra au Cameroun :"),
                                 p("1. Le premier graphique visualise le nombre cumulé de cas dans un diagramme à barres et le nombre cumulé de décès représenté par une ligne :"),
                                 p("   L'utilisateur peut cliquer sur la barre de la région dans le premier graphique ci-dessous pour en savoir plus sur la tendance des cas et des décès dans cette région particulière."),                                 
                                 p("2. Le deuxième graphique visualise le nombre de cas selon la chronologie rapportée dans les régions :"),
                                 p("   L'utilisateur peut survoler le graphique linéaire de la région concernée pour en savoir plus sur l'historique des cas de choléra dans la région. Ce graphique aide à comprendre la tendance des cas de choléra dans chaque région et permet également à l'utilisateur de la comparer avec le graphique sélectionné dans le premier graphique."),
                                 
                                 #Plot - 1
                                 plotlyOutput("drilldown_plot", width = "900px", height = "300px"
                                 ),
                                 
                                 #To go to the previous page from the selected graph
                                 uiOutput("back_button"), 
                                 
                                 #Plot - 2
                                 plotlyOutput("time" , width = "900px", height = "300px")
                        ))
             ),
             
             #---------------- Data tab ---------- UI code --------------------------------
             
             tabPanel(h5("Données"),
                      mainPanel(
                        h3("Données sur l'incidence du choléra au Cameroun"),
                        p("Dans ce panneau, les données brutes utilisées pour créer les graphiques et les diagrammes ont été affichées. Les utilisateurs peuvent visualiser, rechercher, trier et télécharger les données pour une utilisation ultérieure ou une meilleure compréhension."),
                        dataTableOutput(outputId = "Cdatatable"),
                        downloadButton("download_data", "Télécharger les données")
                      )
             ),
             
             #---------------- About tab ---------- UI code ------------------------------
             
             tabPanel(h5("À propos"),
                      h3("Code source"), 
                      p("Le code source pour le traitement du tableau de bord est disponible ici :",
                        tags$a(href='https://github.com/rodekruis/RIPOSTE/tree/main/Shiny%20App%20Cholera', 'Shiny App Cholera')), #to add github link here
                      h3("Source des données"), 
                      div(
                        p("Les frontières administratives du Cameroun ont été obtenues à partir de ",
                          tags$a(href='https://data.humdata.org/dataset/geoboundaries-admin-boundaries-for-cameroon?', 'Cameroon Geoboundaries')),
                        p("Source de données pour la visualisation est le CCOUSP | PHEOCC, Voir",
                          tags$a(href='https://www.ccousp.cm/urgences-sanitaires/cholera/situation-cholera-cameroun/', 'Cholera Situation Reports - Cameroon'),
                        ))
             ),
  )

# Defining breaks in Main tab of the dashboard ---------------------------------

#breaks for cases
breaks = c(0, 50, 100, 500, 1000, 1500)

#Adding color palette to the map, colorBin gets palette information from ColorBrewer2
pal <- colorBin("Reds", domain = merged$cases, bins = breaks)

basemap = leaflet(outline) %>%
  addTiles() %>%
  addProviderTiles(providers$CartoDB.Positron) %>%
  setView(12, 7, zoom = 5.5, options = list) %>%
  addLegend("bottomright",
            pal = pal,
            values = ~merged$cases,
            title="Nombre de Cas",
            opacity = 1
  )

# 3. Define server -------------------------------------------------------------

server <- function(input, output, session) {
  
  #---------------- Cholera tab ---------- server code -------------------------
  
  filtered_data <- reactive({
    #converting input date to date class
    input_date <- as.Date(input$date_slider, format = "%d %b %y")
    # Filter data based on input date
    filter(merged, start_date == input_date)
  })
  
  output$mymap <- renderLeaflet({
    basemap
  })
  
  # Rendering the leaflet map
  observeEvent(input$date_slider, {
    leafletProxy("mymap") %>%
      addTiles()   %>%
      addPolygons(data = filtered_data(),
                  fillColor = ~pal(filtered_data()$cases),
                  fillOpacity = 0.7,
                  color = "grey",
                  smoothFactor = 0.1,
                  stroke = TRUE,
                  weight = 1,
                  label= sprintf("Region: <b>%s</b><br>Nombre de cas: %d<br/>Nombre de décès: %d", filtered_data()$ADM1_FR,filtered_data()$cases,filtered_data()$deaths) %>% lapply(htmltools::HTML),
                  highlightOptions = highlightOptions(color = "black", bringToFront = TRUE, weight = 2)
      )
  })  
  
  #------------ Outbreak Plots tab ---------- server code ---------------------- 
  
  # For Bar and Line Plot tab --------------------------
  # Define a custom color palette with 10 colors for each region - color blind friendly
  custom_palette <- c("#000000", "#2271B2", "#3DB7E9", "#8400CD","#9F0162", "#359B73", "#A40122", "#D55E00", "#E69F00", "#F0E442" )
  
  # Reactive values to track drill-down state
  current_region <- reactiveVal(NULL)

  # Filtered data based on drill-down
  plot_data <- reactive({
    if (is.null(current_region())) {
      #For Level 1 data filtering
      incidence %>%
        group_by(ADM1_FR) %>%
        summarize(cases = sum(cases), deaths = sum(deaths))
    } else {
      #Level 2 data filtering
      incidence %>%
        filter(ADM1_FR == current_region()) %>%
        arrange(start_date)
    }
  })
  
  # Plot 1 ----------------------------------------
  output$drilldown_plot <- renderPlotly({
    
    p <- if (is.null(current_region())) {
      # Level 1 Plot - regions on x-axis
      plot_ly(plot_data(),
              x = ~ADM1_FR, y = ~cases, type = 'bar', name = 'Cas') %>%
        add_trace(y = ~deaths, type = 'scatter', mode = 'lines+markers', name = 'Décès', yaxis = 'y2')
    } else {
      # Level 2 Plot - start_date on x-axis
      plot_ly(plot_data(),
              x = ~start_date, y = ~cases, type = 'bar', name = 'Cas') %>%
        add_trace(y = ~deaths, type = 'scatter', mode = 'lines+markers', name = 'Décès', yaxis = 'y2') %>%
        # Handle date summarization 
        layout(barmode = "group")  # Group bars are easier for comparison
      
    }
    
    #Shared Layout  
    p %>% 
      layout(
        xaxis = list(title = current_region() %||% "Régions du Cameroun"),
        yaxis = list(
          title = 'Nombre de cas',
          range = c(min(plot_data()$cases), 
                    max(plot_data()$cases) + 10)),
        yaxis2 = list(
          overlaying = 'y',
          title = 'Nombre de décès',
          side = 'right',
          range = c(min(plot_data()$deaths),
                    max(plot_data()$deaths) + 10)),
        legend = list(title = list(text = '<b> Legend </b>'))
      ) %>% 
      # Control drill-down behavior
      event_register("plotly_click")  
  })
  
  # Drill-down logic
  observeEvent(event_data("plotly_click"), {
    current_region(event_data("plotly_click")$x)
  })
  
  #Plot - 2 ------------------------------------
  #Report cases overtime - all regions or a specified region if selected
  data_time <- reactive({
    incidence %>%
      count(ADM1_FR, start_date, wt = cases)
  })
  
  #Renders a Plotly plot of cases overtime
  output$time <- renderPlotly ({
    d <- setNames(data_time(), c("color", "x", "y"))
    
    plot_ly(d) %>%
      add_lines(x = ~x, y = ~y, type = 'scatter', mode = 'lines+markers', color= ~color, colors = custom_palette) %>%
      layout(xaxis = list(title = 'Temps', 
                          range=c(min(date_range), max(date_range))),
             yaxis = list(title = 'Nombre de cas'),
             legend = list(title = '<b> Légende </b>')
      ) 
  })
  
  # Back Button UI ---------------------------
  output$back_button <- renderUI({
    if (!is.null(current_region())) {
      actionButton("back", "Retour à Régions")
    }
  })
  
  # Reset drill-down on back button click
  observeEvent(input$back, {
    current_region(NULL)
  })
  
  #---------------- Data tab ---------- server code ----------------------------
  
  # Create reactive data frame
  variables_selected <- reactive({
    incidence %>% select(input$selected_var)
  })
  
  # Create data table
  output$Cdatatable <- DT::renderDataTable(
    datatable(
      data = incidence , #%>% select(input$selected_var),
      options = list(pageLength = 10),
      colnames = c('ID' = 1, 'Date de déclaration'=2, 'Nom de la région'=3, 'Nombre de cas'=4,'Nombre de décès'=5)
    )
  )
  
  # Download file
  output$download_data <- downloadHandler(
    filename = function() {
      paste0("Choleradata.csv")
    },
    content = function(file) {
      write_csv(incidence %>% select(input$selected_var), file)
    }
  )
}

# 4. Launching the Shiny app  --------------------------------------------------

shinyApp(ui, server)
