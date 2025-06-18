package backend.backend.controller;

import backend.backend.dto.InferenceRequest;
import backend.backend.entity.*;
import backend.backend.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;
import org.springframework.transaction.annotation.Transactional;

@Controller
@RequestMapping("/patients")
public class PatientController {
    
    @Autowired
    private PatientService patientService;
    
    @GetMapping
    public String listPatients(@AuthenticationPrincipal Doctor currentDoctor, Model model) {
        model.addAttribute("patients", patientService.getPatientsByDoctor(currentDoctor.getId()));
        return "patients/list";
    }
    
    @GetMapping("/new")
    public String showNewPatientForm(Model model) {
        model.addAttribute("patient", new Patient());
        return "patients/form";
    }
    
    @PostMapping("/save")
    public String savePatient(@ModelAttribute Patient patient, 
                            @AuthenticationPrincipal Doctor currentDoctor,
                            RedirectAttributes redirectAttributes) {
        try {
            patientService.savePatient(patient, currentDoctor);
            redirectAttributes.addFlashAttribute("success", "Paziente salvato con successo");
            return "redirect:/patients";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", e.getMessage());
            return "redirect:/patients/new";
        }
    }
    
    @GetMapping("/{id}")
    @Transactional(readOnly = true)
    public String viewPatient(@PathVariable Long id, 
                            @AuthenticationPrincipal Doctor currentDoctor,
                            Model model) {
        Patient patient = patientService.getPatientById(id)
                .orElseThrow(() -> new RuntimeException("Paziente non trovato"));
        
        // Verifica che il paziente appartenga al medico corrente
        if (!patient.getDoctor().getId().equals(currentDoctor.getId())) {
            throw new RuntimeException("Non autorizzato a visualizzare questo paziente");
        }
        
        model.addAttribute("patient", patient);
        model.addAttribute("medicalRecords", patientService.getPatientMedicalRecords(id));
        return "patients/view";
    }
    
    @GetMapping("/{id}/medical-record/new")
    public String showNewMedicalRecordForm(@PathVariable Long id, 
                                         @AuthenticationPrincipal Doctor currentDoctor,
                                         Model model) {
        Patient patient = patientService.getPatientById(id)
                .orElseThrow(() -> new RuntimeException("Paziente non trovato"));
        
        // Verifica che il paziente appartenga al medico corrente
        if (!patient.getDoctor().getId().equals(currentDoctor.getId())) {
            throw new RuntimeException("Non autorizzato a modificare questo paziente");
        }
        
        model.addAttribute("patient", patient);
        model.addAttribute("medicalRecord", new MedicalRecord());
        model.addAttribute("inferenceRequest", new InferenceRequest());
        return "patients/medical-record-form";
    }
    
    @PostMapping("/{id}/medical-record/save")
    public String saveMedicalRecord(@PathVariable Long id,
                                   @ModelAttribute MedicalRecord medicalRecord,
                                   @ModelAttribute InferenceRequest inferenceRequest,
                                   @AuthenticationPrincipal Doctor currentDoctor,
                                   RedirectAttributes redirectAttributes) {
        try {
            Patient patient = patientService.getPatientById(id)
                    .orElseThrow(() -> new RuntimeException("Paziente non trovato"));
            
            // Verifica che il paziente appartenga al medico corrente
            if (!patient.getDoctor().getId().equals(currentDoctor.getId())) {
                throw new RuntimeException("Non autorizzato a modificare questo paziente");
            }
            
            patientService.createMedicalRecordWithPrediction(patient, medicalRecord, inferenceRequest);
            redirectAttributes.addFlashAttribute("success", "Record medico salvato e predizione effettuata");
            return "redirect:/patients/" + id;
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Errore: " + e.getMessage());
            return "redirect:/patients/" + id + "/medical-record/new";
        }
    }
    
    @GetMapping("/predictions/{predictionId}/review")
    @Transactional(readOnly = true)
    public String showReviewForm(@PathVariable Long predictionId, 
                               @AuthenticationPrincipal Doctor currentDoctor,
                               Model model) {
        Prediction prediction = patientService.getPredictionById(predictionId)
                .orElseThrow(() -> new RuntimeException("Predizione non trovata"));
        
        // Verifica che la predizione appartenga a un paziente del medico corrente
        if (!prediction.getMedicalRecord().getPatient().getDoctor().getId().equals(currentDoctor.getId())) {
            throw new RuntimeException("Non autorizzato a rivedere questa predizione");
        }
        
        model.addAttribute("prediction", prediction);
        return "patients/review-form";
    }
    
    @PostMapping("/predictions/{predictionId}/review")
    public String submitReview(@PathVariable Long predictionId,
                              @RequestParam String reviewNotes,
                              @RequestParam Boolean confirmedDiagnosis,
                              @AuthenticationPrincipal Doctor currentDoctor,
                              RedirectAttributes redirectAttributes) {
        try {
            Prediction prediction = patientService.getPredictionById(predictionId)
                    .orElseThrow(() -> new RuntimeException("Predizione non trovata"));
            
            // Verifica che la predizione appartenga a un paziente del medico corrente
            if (!prediction.getMedicalRecord().getPatient().getDoctor().getId().equals(currentDoctor.getId())) {
                throw new RuntimeException("Non autorizzato a rivedere questa predizione");
            }
            
            patientService.createReview(prediction, currentDoctor, reviewNotes, confirmedDiagnosis);
            redirectAttributes.addFlashAttribute("success", "Revisione salvata con successo");
            return "redirect:/dashboard";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Errore: " + e.getMessage());
            return "redirect:/patients/predictions/" + predictionId + "/review";
        }
    }
    
    @PostMapping("/{id}/delete")
    @Transactional
    public String deletePatient(@PathVariable Long id,
                              @AuthenticationPrincipal Doctor currentDoctor,
                              RedirectAttributes redirectAttributes) {
        try {
            Patient patient = patientService.getPatientById(id)
                    .orElseThrow(() -> new RuntimeException("Paziente non trovato"));
            
            // Verifica che il paziente appartenga al medico corrente
            if (!patient.getDoctor().getId().equals(currentDoctor.getId())) {
                throw new RuntimeException("Non autorizzato a cancellare questo paziente");
            }
            
            patientService.deletePatient(id);
            redirectAttributes.addFlashAttribute("success", "Paziente cancellato con successo");
            return "redirect:/patients";
        } catch (Exception e) {
            redirectAttributes.addFlashAttribute("error", "Errore durante la cancellazione: " + e.getMessage());
            return "redirect:/patients";
        }
    }
} 