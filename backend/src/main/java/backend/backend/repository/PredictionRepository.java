package backend.backend.repository;

import backend.backend.entity.Prediction;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;
import java.util.List;
import java.util.Optional;

@Repository
public interface PredictionRepository extends JpaRepository<Prediction, Long> {
    @Query("SELECT p FROM Prediction p " +
           "JOIN FETCH p.medicalRecord mr " +
           "JOIN FETCH mr.patient " +
           "WHERE p.isConfirmedByDoctor = false " +
           "ORDER BY p.predictionDate DESC")
    List<Prediction> findByIsConfirmedByDoctorFalseOrderByPredictionDateDesc();
    
    @Query("SELECT p FROM Prediction p " +
           "JOIN FETCH p.medicalRecord mr " +
           "JOIN FETCH mr.patient pat " +
           "WHERE p.isConfirmedByDoctor = false " +
           "AND pat.doctor.id = :doctorId " +
           "ORDER BY p.predictionDate DESC")
    List<Prediction> findPendingPredictionsByDoctorId(@Param("doctorId") Long doctorId);
    
    @Query("SELECT p FROM Prediction p " +
           "JOIN FETCH p.medicalRecord mr " +
           "JOIN FETCH mr.patient " +
           "WHERE p.id = :id")
    Optional<Prediction> findByIdWithRelations(@Param("id") Long id);
}