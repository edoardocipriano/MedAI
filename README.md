# MedAI - Applicazione Web per Predizione Diabete

Applicazione web completa per medici che permette di gestire pazienti e utilizzare un modello di intelligenza artificiale per la predizione del diabete. Il sistema combina un backend Spring Boot con un servizio di inferenza Python basato su PyTorch.

## Funzionalità Principali

- **Autenticazione Medici**: Sistema di registrazione e login sicuro con password criptate
- **Gestione Pazienti**: Interfaccia completa per creare, visualizzare e gestire i pazienti
- **Cartelle Cliniche**: Registrazione dettagliata di sintomi, esami del sangue e note mediche
- **Predizioni AI**: Integrazione con modello di deep learning per predizione del rischio diabete
- **Sistema di Revisione**: I medici possono confermare o correggere le predizioni del modello
- **Dashboard Interattiva**: Visualizzazione statistiche, predizioni in attesa e attività recenti

## Architettura del Sistema

Il progetto è composto da tre componenti principali:

1. **Backend Spring Boot** - API REST e interfaccia web
2. **Servizio di Inferenza Python** - API FastAPI per le predizioni ML
3. **Modulo di Training** - Script Python per addestrare il modello

## Struttura Dettagliata del Progetto

### 📁 Backend Spring Boot (`/backend`)

#### 🔹 Main Application
- **`BackendApplication.java`**: Classe principale che avvia l'applicazione Spring Boot con annotazione `@SpringBootApplication`

#### 🔹 Controllers (`/controller`)

- **`LoginController.java`**: Gestisce autenticazione e registrazione
  - `GET /login`: Mostra pagina di login
  - `GET /register`: Mostra form di registrazione
  - `POST /register`: Processa registrazione nuovo medico con validazione email e criptazione password

- **`PatientController.java`**: Gestisce tutte le operazioni sui pazienti
  - `GET /patients`: Lista pazienti del medico autenticato
  - `GET /patients/new`: Form creazione nuovo paziente
  - `POST /patients/save`: Salva nuovo paziente
  - `GET /patients/{id}`: Visualizza dettagli paziente con cartelle cliniche
  - `GET /patients/{id}/medical-record/new`: Form nuova cartella clinica
  - `POST /patients/{id}/medical-record/save`: Salva cartella e richiede predizione AI
  - `GET /patients/predictions/{id}/review`: Form revisione predizione
  - `POST /patients/predictions/{id}/review`: Salva revisione medico

- **`DashboardController.java`**: Dashboard principale del medico
  - `GET /dashboard`: Mostra statistiche, predizioni pendenti e attività recenti
  - `GET /`: Redirect automatico alla dashboard

#### 🔹 Entities (`/entity`)

- **`Doctor.java`**: Entità medico (implementa `UserDetails` per Spring Security)
  - Campi: id, firstName, lastName, specialization, email, password
  - Relazioni: OneToMany con Patient e Review
  - Metodi: gestione autenticazione Spring Security

- **`Patient.java`**: Entità paziente
  - Campi: id, firstName, lastName, dateOfBirth, gender, fiscalCode, createdAt
  - Relazioni: ManyToOne con Doctor, OneToMany con MedicalRecord
  - Annotazioni: @PrePersist per timestamp automatico

- **`MedicalRecord.java`**: Cartella clinica del paziente
  - Campi: id, recordDate, symptoms, notes, bloodTests, createdAt
  - Relazioni: ManyToOne con Patient, OneToOne con Prediction
  - Gestione: dati clinici e sintomi per la predizione

- **`Prediction.java`**: Predizione AI sul diabete
  - Campi: id, predictionDate, result, probability, riskLevel
  - Relazioni: OneToOne con MedicalRecord, OneToOne con Review
  - Calcolo: livello di rischio basato sulla probabilità

- **`Review.java`**: Revisione medica della predizione
  - Campi: id, reviewDate, notes, confirmedDiagnosis
  - Relazioni: ManyToOne con Doctor, OneToOne con Prediction
  - Scopo: validazione umana delle predizioni AI

#### 🔹 Repositories (`/repository`)

- **`DoctorRepository.java`**: Query per medici
  - `findByEmail()`: per autenticazione
  - `existsByEmail()`: validazione registrazione

- **`PatientRepository.java`**: Query per pazienti
  - `findByDoctorId()`: pazienti di un medico
  - `existsByFiscalCode()`: validazione codice fiscale

- **`MedicalRecordRepository.java`**: Query cartelle cliniche
  - `findByPatientIdOrderByRecordDateDesc()`: storia clinica paziente

- **`PredictionRepository.java`**: Query predizioni
  - `findByReviewIsNullOrderByPredictionDateDesc()`: predizioni da revisionare
  - `countByReviewIsNull()`: conteggio predizioni pendenti

- **`ReviewRepository.java`**: Query revisioni
  - `findTop10ByDoctorIdOrderByReviewDateDesc()`: ultime revisioni

#### 🔹 Services (`/service`)

- **`CustomUserDetailsService.java`**: Integrazione Spring Security
  - Implementa `UserDetailsService`
  - Carica utenti dal database per autenticazione

- **`MLInferenceService.java`**: Client per servizio ML Python
  - Usa `RestTemplate` per chiamate HTTP
  - Endpoint: `POST /predict` al servizio Python
  - Gestione errori e timeout

- **`PatientService.java`**: Logica business principale
  - Gestione completa pazienti e cartelle cliniche
  - Orchestrazione predizioni: salva record → chiama ML → salva risultato
  - Validazioni e controlli di sicurezza

#### 🔹 DTOs (`/dto`)

- **`InferenceRequest.java`**: Request per servizio ML
  - Campi: age, gender, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level
  - Mapping da MedicalRecord a formato ML

- **`InferenceResponse.java`**: Response dal servizio ML
  - Campi: diabetes (Yes/No), probability (0-1)
  - Conversione in entità Prediction

#### 🔹 Configuration (`/config`)

- **`AppConfig.java`**: Configurazioni generali
  - Bean `RestTemplate` per chiamate HTTP
  - Configurazioni timeout e connessioni

- **`SecurityConfig.java`**: Configurazione Spring Security
  - Autenticazione form-based
  - Protezione endpoint (login/register pubblici)
  - BCrypt per password
  - Gestione sessioni e logout

#### 🔹 Templates (`/templates`)

- **`login.html`**: Pagina di login con form e link registrazione
- **`register.html`**: Form registrazione medico con validazioni
- **`dashboard.html`**: Dashboard con statistiche, grafici e liste
- **`patients/list.html`**: Tabella pazienti con azioni
- **`patients/form.html`**: Form creazione/modifica paziente
- **`patients/view.html`**: Dettagli paziente e storia clinica
- **`patients/medical-record-form.html`**: Form cartella clinica con campi per ML
- **`patients/review-form.html`**: Form revisione predizione con dettagli

#### 🔹 Static Resources (`/static`)

- **`css/style.css`**: Stili custom per UI moderna
  - Design responsive
  - Tema medicale professionale
  - Animazioni e transizioni

#### 🔹 Configuration Files

- **`application.properties`**: Configurazioni Spring Boot
  ```properties
  # Database PostgreSQL
  spring.datasource.url=jdbc:postgresql://localhost:5432/medai_db
  spring.datasource.username=postgres
  spring.datasource.password=yourpassword
  
  # JPA/Hibernate
  spring.jpa.hibernate.ddl-auto=update
  spring.jpa.show-sql=true
  
  # ML Service
  ml.service.url=http://localhost:8000
  ```

- **`pom.xml`**: Dipendenze Maven
  - Spring Boot Starter Web, Data JPA, Security, Thymeleaf
  - PostgreSQL Driver
  - Spring Boot DevTools

### 📁 Servizio Inferenza Python (`/model_inference`)

- **`inference.py`**: API FastAPI per predizioni
  - Endpoint `POST /predict`: riceve dati paziente, restituisce predizione
  - Preprocessing: normalizzazione features, encoding categoriche
  - Caricamento modello PyTorch salvato
  - Threshold ottimizzato (0.35) per bilanciare sensibilità/specificità

- **`schemas.py`**: Modelli Pydantic per validazione
  - `InputData`: schema dati in input (matching InferenceRequest)
  - `OutputData`: schema risposta (diabetes: Yes/No, probability)

### 📁 Training Modello (`/model_training`)

- **`main.py`**: Script principale per training
  - Training loop con validazione
  - Ottimizzazione iperparametri
  - Generazione metriche (accuracy, precision, recall, F1)
  - Salvataggio best model basato su recall
  - Creazione grafici performance

- **`model.py`**: Architettura rete neurale
  - Multi-layer perceptron con dropout
  - Input: 10 features
  - Hidden layers: [64, 32, 16] neuroni
  - Activation: ReLU + Batch Normalization
  - Output: 1 neurone (probabilità diabete)

- **`data_utils.py`**: Preprocessing dataset
  - Caricamento CSV diabetes dataset
  - Gestione valori mancanti
  - Encoding variabili categoriche
  - Scaling features con MinMaxScaler
  - Split train/test stratificato
  - Creazione DataLoader PyTorch

- **`diabetes_prediction_dataset.csv`**: Dataset training
  - 100k+ record pazienti
  - Features: dati demografici, clinici, stile di vita
  - Target: diagnosi diabete (0/1)

- **`diabetes_model.pth`**: Modello trained salvato
  - Pesi rete neurale ottimizzati
  - Pronto per inferenza

#### 📁 Sottocartelle

- **`/graphs`**: Visualizzazioni performance
  - `confusion_matrix.png`: Matrice confusione
  - `roc_curve.png`: Curva ROC con AUC
  - `training_history.png`: Loss e accuracy durante training

- **`/saved`**: Artefatti preprocessing
  - `column_transformer.pkl`: Scaler salvato per consistency

## Workflow Completo

1. **Registrazione Medico**: Nuovo medico si registra → password criptata → salvata in DB
2. **Login**: Autenticazione Spring Security → sessione → redirect dashboard
3. **Creazione Paziente**: Form dati anagrafici → validazione → salvataggio con associazione medico
4. **Aggiunta Cartella Clinica**: 
   - Medico inserisce sintomi e valori esami
   - Submit → PatientService prepara InferenceRequest
   - Chiamata REST a servizio Python
   - Python preprocessa dati → inferenza modello → ritorna predizione
   - Salvataggio predizione in DB
5. **Review Predizione**: 
   - Dashboard mostra predizioni pendenti
   - Medico apre dettagli → valuta → conferma o corregge
   - Feedback salvato per future analisi

## Tecnologie e Framework

### Backend
- **Spring Boot 3.5.0**: Framework principale
- **Spring Security**: Autenticazione e autorizzazione
- **Spring Data JPA**: ORM e gestione database
- **Thymeleaf**: Template engine per viste
- **PostgreSQL**: Database relazionale
- **Maven**: Build e dependency management

### ML/AI
- **Python 3.8+**: Linguaggio per ML
- **PyTorch**: Framework deep learning
- **FastAPI**: API REST per inferenza
- **scikit-learn**: Preprocessing e metriche
- **NumPy/Pandas**: Manipolazione dati

### Frontend
- **Bootstrap CSS**: Framework UI (customizzato)
- **JavaScript**: Interattività client-side
- **Chart.js**: Grafici dashboard

## Sicurezza

- Password criptate con BCrypt
- Sessioni gestite da Spring Security
- Validazione input lato server
- Protezione CSRF
- Controlli autorizzazione per-richiesta

## Performance e Scalabilità

- Lazy loading relazioni JPA
- Caching predizioni
- Connection pooling database
- Servizio ML separato (scalabile indipendentemente)
- Possibilità deploy containerizzato

## Prerequisiti Sistema

- Java 21
- PostgreSQL 12+
- Python 3.8+
- Maven 3.6+
- 4GB RAM minimo
- 1GB spazio disco

## Istruzioni Installazione

### 1. Setup Database
```sql
CREATE DATABASE medai_db;
CREATE USER medai_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE medai_db TO medai_user;
```

### 2. Configurazione Backend
Aggiorna `application.properties` con le tue credenziali database.

### 3. Avvio Servizio ML
```bash
cd model_inference
pip install -r requirements.txt
python inference.py
```

### 4. Avvio Backend
```bash
cd backend
./mvnw spring-boot:run
```

### 5. Accesso Applicazione
- URL: http://localhost:8080
- Registra account medico
- Inizia a utilizzare il sistema

## Testing

Il progetto include:
- Unit test per services
- Integration test per controllers
- Test API per servizio ML
- Validazione modello su test set

## Deployment

Per deployment in produzione:
1. Build JAR: `./mvnw clean package`
2. Containerizza con Docker
3. Configura reverse proxy (nginx)
4. Setup SSL/TLS
5. Monitoring con Spring Actuator

## Contribuire

1. Fork repository
2. Crea feature branch
3. Commit modifiche
4. Push e apri Pull Request

## Licenza

Progetto universitario - uso educativo

## Contatti

Per domande o supporto, contattare il team di sviluppo. 