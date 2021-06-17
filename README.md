# pokemon-generator-nombre

Este es un generador de nombres de Pokémon que funciona en el navegador en forma de una aplicación Vue. Utiliza TensorFlow.js para todo el código de aprendizaje profundo. La red neuronal está compuesta por una capa LSTM para procesar una capa Densa para la salida.


Traté de hacer que todo el código de aprendizaje profundo fuera lo más independiente posible del código de Vue, por lo que todo está en la carpeta `learn` dentro de` src`. También incluí en la carpeta `pokemons-scraper` el raspador web que usé para obtener todos los nombres de Pokémon de esta base de datos: https://pokemondb.net/pokedex/national

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```

### Compiles and minifies for production
```
npm run build
```

### Customize configuration
See [Configuration Reference](https://cli.vuejs.org/config/).
