package backend.backend.controller;

import backend.backend.entity.Doctor;
import backend.backend.service.PatientService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.stereotype.Controller;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class DashboardController {
    
    @Autowired
    private PatientService patientService;
    
    @GetMapping("/")
    public String redirectToDashboard() {
        return "redirect:/dashboard";
    }
    
    @GetMapping("/dashboard")
    @Transactional(readOnly = true)
    public String showDashboard(@AuthenticationPrincipal Doctor currentDoctor, Model model) {
        model.addAttribute("doctor", currentDoctor);
        model.addAttribute("patients", patientService.getPatientsByDoctor(currentDoctor.getId()));
        model.addAttribute("pendingPredictions", patientService.getPendingPredictionsByDoctor(currentDoctor.getId()));
        model.addAttribute("doctorReviews", patientService.getDoctorReviews(currentDoctor.getId()));
        
        return "dashboard";
    }
} 