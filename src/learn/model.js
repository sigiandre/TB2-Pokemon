const tf = require("@tensorflow/tfjs")
window.tf = tf;

class Generator {
    constructor() {}

    // Carga un modelo cuando carga la página por primera vez
    async loadModel() {
        this.longestLength = 12
        this.vocabSize = 28
        this.vocab = {
            " ": 0, ".": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8, "h": 9, "i": 10, "j": 11, "k": 12, "l": 13, "m": 14, "n": 15, "o": 16, "p": 17, "q": 18, "r": 19, "s": 20, "t": 21, "u": 22, "v": 23, "w": 24, "x": 25, "y": 26, "z": 27
        }
        this.vocabInverse = {
            0: " ", 1: ".", 2: "a", 3: "b", 4: "c", 5: "d", 6: "e", 7: "f", 8: "g", 9: "h", 10: "i", 11: "j", 12: "k", 13: "l", 14: "m", 15: "n", 16: "o", 17: "p", 18: "q", 19: "r", 20: "s", 21: "t", 22: "u", 23: "v", 24: "w", 25: "x", 26: "y", 27: "z"
        }
        this.model = await tf.loadLayersModel("/my-model-trained-300.json")
    }

    async createModelAndTrain(dataset, log) {
        this.preprocess(dataset, log) // Crea conjuntos de vocabulario y entrenamiento
        this.createModel(log) // Crea el modelo TFjs a partir de dimensiones anteriores.
        // Entrena el modelo
        return this.model.fit(this.trainX, this.trainY, { 
            batchSize: 64,
            epochs: 301,
            callbacks: [
                new tf.CustomCallback({
                    onEpochEnd: async(epoch, logs) => {
                        log("Época " + epoch + "/300 terminada")
                        if (epoch % 25 == 0) {
                            // Registra 5 nombres cada 25 épocas de entrenamiento
                            log("Nombre de la prueba: " + this.predict(""))
                            log("Nombre de la prueba: " + this.predict(""))
                            log("Nombre de la prueba: " + this.predict(""))
                            log("Nombre de la prueba: " + this.predict(""))
                            log("Nombre de la prueba: " + this.predict(""))
                            log("===========================")
                        }
                    }
                })
            ],
            yieldEvery: 5000
        })
    }

    preprocess(dataset, log) {
        let names = dataset.toLowerCase().split("\n")
        let longestName = ""
        // Obtiene el nombre más largo, por lo que todos los demás nombres son más pequeños
        for (let i = 0; i < names.length; i++) {
            names[i] = names[i] + "."
            if (names[i].split("").length > longestName.split("").length) {
                longestName = names[i]
            }
        }
        log("Nombre más largo: " + longestName)

        // Crea vocabulario de todos los caracteres individuales que se encuentran en el conjunto de datos
        let uniqueChars = String.prototype.concat(...new Set(names.join(""))).split("").sort()
        this.vocab = {}
        this.vocabInverse = {}
        for (let i = 0; i < uniqueChars.length; i++) {
            const char = uniqueChars[i];
            this.vocab[char] = i
            this.vocabInverse[i] = char
        }
        log("Vocabulario: " + Object.keys(this.vocab))

        this.vocabSize = Object.keys(this.vocab).length
        this.longestLength = longestName.length - 1

        // Crea matrices de vectores de entrada y salida convirtiendo los nombres en
        // one-hots de tamaño igual al tamaño más largo (lleno de ceros en el resto)
        let arrayNamesIn = []
        let arrayNamesOut = []
        names.forEach(name => {
            const wordIn = name.split("").slice(0, name.split("").length - 1)
            const wordOut = name.split("").slice(1, name.split("").length)
            arrayNamesIn.push(this.convertToOneHot(wordIn, this.vocab))
            arrayNamesOut.push(this.convertToOneHot(wordOut, this.vocab))
        });

        log("Número de muestras: " + arrayNamesIn.length)

        // Convierte matrices en tensores de entrenamiento de entrada y salida
        this.trainX = tf.tensor3d(arrayNamesIn)
        this.trainY = tf.tensor3d(arrayNamesOut)
    }

    createModel(log) {
        // El modelo es solo un tipo secuencial con dos capas (lstm y denso)
        // para la salida
        this.model = tf.sequential()
        this.model.add(tf.layers.lstm({
            units: 128,
            inputShape: [this.longestLength, this.vocabSize],
            returnSequences: true
        }))
        this.model.add(tf.layers.dense({ 
            units: this.vocabSize, activation: "softmax" 
        }))

        this.model.compile({ 
            loss: "categoricalCrossentropy",
            optimizer: "adam"
        })
        
        this.model.summary()
    
        log("Resumen impreso en la consola")
    }

    // Muestra una palabra completa con el modelo. wordBeginning puede estar vacío, por lo que es un
    // palabra completamente nueva, o puede tener algunos caracteres para completarla.
    predict(wordBeginning) {
        let newName = wordBeginning.split("")
        let x = tf.zeros([1, this.longestLength, this.vocabSize]) // Crea un tensor de ceros del tamaño de la palabra
        let xArr = x.arraySync()
        // Rellena el tensor con puntos únicos correspondientes a los caracteres de la palabra Principio
        for (let i = 0; i < wordBeginning.split("").length; i++) {
            const char = wordBeginning.split("")[i];
            xArr[0][i][this.vocab[char]] = 1
        }
        x = tf.tensor3d(xArr)
        let end = false
        let i = wordBeginning.split("").length

        // Itera para probar cada nuevo personaje con el nuevo nombre
        while (!end) {
            let newCharacter
            if (i == this.longestLength) {
                // Si es demasiado largo, termínelo con un punto.
                newCharacter = "."
                newName.push(newCharacter)
                end = true
            } else {
                // Predice nuevo carácter de todas las anteriores.
                let probs = this.model.predict(x).arraySync()[0][i]
                let probsSum = 0
                probs.forEach(prob => {
                    probsSum += prob
                });
                for (let i = 0; i < probs.length; i++) {
                    probs[i] = probs[i] / probsSum;
                }

                let charId = randomChoices(probs, 1)[0]
                newCharacter = this.vocabInverse[charId]
                newName.push(newCharacter)
                let xArr = x.arraySync()
                xArr[0][i][charId] = 1
                x = tf.tensor3d(xArr)

                if (newCharacter == ".") {
                    end = true
                }

                i += 1
            }
        }

        return newName.join("")
    }

    convertToOneHot(word, vocab) {
        let encodedWord = []
        // Convierte en one-hot todos los personajes del mundo
        word.forEach(char => {
            let charOneHot = []
            for (let i = 0; i < Object.keys(vocab).length; i++) {
                if (vocab[char] == i) {
                    charOneHot.push(1)
                } else {
                    charOneHot.push(0)
                }                
            }
            encodedWord.push(charOneHot)
        });

        // Y convertir a una matriz de ceros todo lo que queda por más tiempo
        for (let i = word.length; i < this.longestLength; i++) {
            let charOneHot = []
            for (let i = 0; i < Object.keys(vocab).length; i++) {
                charOneHot.push(0)
            }
            encodedWord.push(charOneHot)
        }

        return encodedWord
    }
}

function randomChoice(p) {
    let rnd = p.reduce( (a, b) => a + b ) * Math.random();
    return p.findIndex( a => (rnd -= a) < 0 );
}

function randomChoices(p, count) {
    return Array.from(Array(count), randomChoice.bind(null, p));
}

export default Generator