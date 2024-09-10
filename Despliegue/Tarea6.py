import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.graph_objs as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pickle

file_name = "despliegue/xgboost_reg.pkl"
best_model = pickle.load(open(file_name, "rb"))

app = dash.Dash(__name__)

app.layout = html.Div([

    html.Div([
        html.H1("Predicción de Rentas de Bicicletas y Análisis Financiero", style={'textAlign': 'center', 'color': '#333'}),

        html.Label('Número de bicicletas disponibles:'),
        dcc.Input(id='num-bikes-input', type='number', value=300, min=1, 
                  style={'marginBottom': '10px', 'display': 'block'}),  # Estilo para que se coloque uno debajo de otro

        html.Label('Fecha:'),
        dcc.DatePickerSingle(
            id='date-input',
            date=pd.to_datetime('2017-12-01').date(),  # Valor inicial
            display_format='YYYY-MM-DD',
            min_date_allowed=pd.to_datetime('2017-12-01').date(),  # Fecha mínima permitida
            max_date_allowed=pd.to_datetime('2018-11-30').date(),  # Fecha máxima permitida
            style={'marginBottom': '10px', 'display': 'block'}  # Estilo para que se coloque uno debajo de otro
        ),

        html.Label('Hora del día (0-23):'),
        dcc.Input(id='hour-input', type='number', value=12, min=0, max=23, 
                  style={'marginBottom': '10px', 'display': 'block'}),  # Estilo para que se coloque uno debajo de otro

        
        html.Label('Temperatura (C):'),
        dcc.Slider(
            id='temp-input',
            min=-20,
            max=50,
            step=1,
            value=20,
            marks={i: str(i) for i in range(-20, 51, 10)},  # Marcas de cada 10 grados
            tooltip={"placement": "bottom", "always_visible": True},
        ),

        html.Label('Humedad (%):'),
        dcc.Slider(
            id='humidity-input',
            min=0,
            max=100,
            step=1,
            value=50,
            marks={i: str(i) for i in range(0, 101, 10)},  # Marcas de cada 10%
            tooltip={"placement": "bottom", "always_visible": True},
        ),

      
        html.Label('Radiación Solar (MJ/m2):'),
        dcc.Slider(
            id='solar-input',
            min=0,
            max=30,
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, 31, 5)},  # Marcas de cada 5 MJ/m2
            tooltip={"placement": "bottom", "always_visible": True},
        ),

        html.Label('Precipitación (mm):'),
        dcc.Slider(
            id='rainfall-input',
            min=0,
            max=200,
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, 201, 50)},  # Marcas de cada 50 mm
            tooltip={"placement": "bottom", "always_visible": True},
        ),

        html.Label('Nieve (cm):'),
        dcc.Slider(
            id='snowfall-input',
            min=0,
            max=100,
            step=1,
            value=0,
            marks={i: str(i) for i in range(0, 101, 20)},  # Marcas de cada 20 cm
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        

        html.Label('¿Es un día laboral? (Functioning Day):'),
        dcc.RadioItems(
            id='functioning-day-radio',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=1,  
            style={'marginBottom': '10px'}
        ),

        html.Label('¿Es verano (Seasons_Summer)?:'),
        dcc.RadioItems(
            id='season-summer-radio',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,  
            style={'marginBottom': '10px'}
        ),

        html.Label('¿Es festivo (Holiday)?:'),
        dcc.RadioItems(
            id='holiday-radio',
            options=[
                {'label': 'Sí', 'value': 1},
                {'label': 'No', 'value': 0}
            ],
            value=0,  
            style={'marginBottom': '10px'}
        ),
     

        html.Div([
            html.Button('Predecir Bicicletas', id='predict-button', n_clicks=0,
                        style={'backgroundColor': '#ADD8E6', 'color': 'black', 'border': 'none', 
                               'padding': '10px 20px', 'textAlign': 'center', 'fontSize': '16px'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),

        html.Div(id='prediction-output', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '20px'}),
        html.Div(id='financial-output', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '20px'}),
        dcc.Graph(id='financial-graph'), 
        dcc.Graph(id='cost-pie-chart'),  
        dcc.Graph(id='daily-rentals-graph'), 
        html.Div(id='daily-financial-summary', style={'marginTop': '30px', 'textAlign': 'center', 'fontSize': '20px'}) 
    ], style={'padding': '40px', 'backgroundColor': '#f9f9f9', 'borderRadius': '15px', 
              'width': '70%', 'margin': 'auto', 'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'})
])

# Callback para predicción y análisis financiero
@app.callback(        
    [Output('prediction-output', 'children'),
     Output('financial-output', 'children'),
     Output('financial-graph', 'figure'),
     Output('cost-pie-chart', 'figure'),
     Output('daily-rentals-graph', 'figure'),
     Output('daily-financial-summary', 'children')], 
    Input('predict-button', 'n_clicks'),
    State('num-bikes-input', 'value'),
    State('hour-input', 'value'),
    State('temp-input', 'value'),
    State('humidity-input', 'value'),
    State('solar-input', 'value'),  
    State('rainfall-input', 'value'), 
    State('snowfall-input', 'value'),  
    State('functioning-day-radio', 'value'),  
    State('season-summer-radio', 'value'),
    State('holiday-radio', 'value'),
    State('date-input', 'date') 
    
)
def update_prediction_and_financial_analysis(n_clicks, num_bikes, hour, temp, humidity, solar_radiation, rainfall, snowfall, functioning_day, season_summer, holiday, selected_date):
    if n_clicks > 0:  
        global best_model  

        fecha_minima=pd.to_datetime('2017-12-01').toordinal()
        fecha_maxima=pd.to_datetime('2018-11-30').toordinal()

        # Convertir la fecha seleccionada a formato datetime y luego a ordinal
        selected_date_ordinal = pd.to_datetime(selected_date).toordinal()
        
        # Escalar la fecha entre 0 y 1 usando los valores fecha_minima y fecha_maxima
        date_scaled = (selected_date_ordinal - fecha_minima) / (fecha_maxima - fecha_minima)
        
        data_xgb = data[['Rented Bike Count',"Functioning Day", "Hour", "Temperature(C)", "Rainfall(mm)", "Solar Radiation (MJ/m2)", "Humidity(%)", "Date_scaled", "Holiday", "Seasons_Summer", "Snowfall (cm)"]]

        # Preparar los datos de entrada para la predicción
        input_data = pd.DataFrame({
            'Functioning Day': [functioning_day],
            'Hour': [hour], 
            'Temperature(C)': [temp],
            'Rainfall(mm)': [rainfall],  
            'Solar Radiation (MJ/m2)': [solar_radiation],  
            'Humidity(%)': [humidity],
            'Date_scaled': [date_scaled],
            'Holiday': [holiday],
            'Seasons_Summer': [season_summer],
            'Snowfall (cm)': [snowfall],  
        })

        try:
            prediction = best_model.predict(input_data)[0]
            
            # Cálculo financiero basado en la predicción
            rental_price_regular = 9000
            maintenance_cost_per_bike_per_day = 3000
            storage_cost_per_unused_bike_per_day = 500

            # Cálculos financieros
            rented_bikes = int(prediction)
            unused_bikes = num_bikes - rented_bikes

            total_revenue = rented_bikes * rental_price_regular
            total_maintenance_cost = num_bikes * maintenance_cost_per_bike_per_day
            total_storage_cost = unused_bikes * storage_cost_per_unused_bike_per_day
            total_costs = total_maintenance_cost + total_storage_cost

            financial_output = f"Ingresos: ${total_revenue} COP | Costos: ${total_costs} COP | Beneficio neto: ${total_revenue - total_costs} COP"

            # Crear el gráfico financiero
            financial_figure = {
                'data': [
                    go.Bar(
                        x=['Ingresos', 'Costos', 'Beneficio Neto'],
                        y=[total_revenue, total_costs, total_revenue - total_costs],
                        marker=dict(color=['green', 'red', 'blue'])
                    )
                ],
                'layout': go.Layout(
                    title='Análisis Financiero por la hora ingresada',
                    xaxis={'title': 'Categoría'},
                    yaxis={'title': 'Valor en COP'},
                    showlegend=False
                )
            }

            # Crear el gráfico de pastel para desglose de costos
            # Prepara los datos
            labels = []
            values = []
            
            if total_maintenance_cost > 0:
                labels.append('Costo de Mantenimiento')
                values.append(total_maintenance_cost)
            
            if total_storage_cost > 0:
                labels.append('Costo de Almacenamiento')
                values.append(total_storage_cost)
            
            # Si no hay datos para mostrar, puedes poner un valor mínimo para asegurar que el gráfico no esté vacío
            if not labels:
                labels = ['Sin Datos']
                values = [0.01]  # Valor mínimo para mostrar algo en el gráfico

            # Crea el gráfico de pastel
            pie_chart_figure = {
                'data': [
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=.3  # Hacer el gráfico de pastel con un agujero en el medio (opcional)
                    )
                ],
                'layout': go.Layout(
                    title='Desglose de Costos'
                )
            }

            # Predicción para cada hora del día (0-23)
            hourly_rentals = []
            for h in range(24):
                input_data['Hour'] = [h]
                hourly_prediction = best_model.predict(input_data)[0]
                hourly_rentals.append(int(hourly_prediction))

            # Crear el gráfico de barras para las bicicletas rentadas en cada hora del día
            daily_rentals_figure = {
            'data': [
                go.Bar(
                    x=list(range(24)),
                    y=hourly_rentals,
                    marker=dict(color='blue')
                )
            ],
            'layout': go.Layout(
                title='Cantidad de bicicletas rentadas al día', 
                xaxis={'title': 'Hora del Día'},
                yaxis={'title': 'Bicicletas Rentadas'},
                showlegend=False
            )
        }

            # Cálculo de ingresos, costos y utilidad neta para el día completo
            total_bikes_rented = sum(hourly_rentals)
            daily_total_revenue = total_bikes_rented * rental_price_regular
            daily_total_storage_cost = max(0, (num_bikes - total_bikes_rented) * storage_cost_per_unused_bike_per_day)
            daily_total_maintenance_cost = max(0, num_bikes * maintenance_cost_per_bike_per_day)
            daily_total_costs = daily_total_maintenance_cost + daily_total_storage_cost
            daily_net_profit = daily_total_revenue - daily_total_costs

            daily_financial_summary = f"Ingresos diarios: ${daily_total_revenue} COP | Costos diarios: ${daily_total_costs} COP | Beneficio neto diario: ${daily_net_profit} COP"

            return (
                html.Div(f'Cantidad estimada de bicicletas rentadas: {int(prediction)}'),
                html.Div(financial_output),
                financial_figure,
                pie_chart_figure,
                daily_rentals_figure,
                html.Div(daily_financial_summary) 
            )

        except Exception as e:
            return html.Div(f"Error al hacer la predicción: {e}"), html.Div(""), go.Figure(), go.Figure(), go.Figure(), html.Div("")

    return html.Div("Haz clic en el botón para predecir la cantidad de bicicletas."), html.Div(""), go.Figure(), go.Figure(), go.Figure(), html.Div("")


if __name__ == '__main__':
    app.run_server(debug=True)
